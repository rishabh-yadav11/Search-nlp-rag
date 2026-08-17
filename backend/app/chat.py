"""Per-user chat conversations stored in SQLite.

Conversations survive restarts and are purged after CHAT_RETENTION_DAYS of
inactivity. Users are identified by an anonymous X-User-Id header (device UUID);
this is not authentication.

The turn pipeline reuses the shared retrieval/rerank/fallback pipeline and
builds a conversation-aware prompt so the model can follow up on prior turns.
"""

import asyncio
import json
import logging
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

import aiosqlite
from fastapi import APIRouter, Header, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.config import config
from app.cost_budget import BudgetExceeded, assert_within_budget, record_cost
from app.llm import LLMResult, LLMUnavailableError, generate_answer, stream_answer

logger = logging.getLogger("chat")

router = APIRouter(prefix="/api/chat", tags=["chat"])

MIN_USER_ID_LEN = 8
MAX_USER_ID_LEN = 128
MAX_CONTENT_LEN = 8000
PREVIEW_LEN = 140

# Module-level store; set by main.lifespan (and by tests).
store: "ChatStore | None" = None


class SessionOut(BaseModel):
    id: str
    title: str
    created_at: float
    updated_at: float
    last_preview: str = ""
    total_cost: float = 0.0


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    sources: list[dict] = []
    created_at: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    latency_ms: float = 0.0


class SessionDetailOut(SessionOut):
    messages: list[MessageOut] = []


class SessionStatsOut(BaseModel):
    sessions: int = 0
    messages: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0


class MessageIn(BaseModel):
    content: str


class TurnOut(BaseModel):
    user: MessageOut
    assistant: MessageOut
    note: str | None = None
    latency_ms: float = 0.0


def _now() -> float:
    return time.time()


def _user_id(x_user_id: str | None) -> str:
    if not x_user_id or len(x_user_id) < MIN_USER_ID_LEN:
        raise HTTPException(status_code=400, detail="X-User-Id header required (min 8 chars)")
    return x_user_id[:MAX_USER_ID_LEN]


class ChatStore:
    """SQLite-backed conversation store. WAL mode + busy_timeout so multiple
    gunicorn workers can read/write concurrently without "database is locked"."""

    def __init__(self, path: str):
        self._path = path
        self._db: aiosqlite.Connection | None = None

    async def connect(self) -> None:
        parent = os.path.dirname(os.path.abspath(self._path))
        os.makedirs(parent, exist_ok=True)
        self._db = await aiosqlite.connect(self._path)
        self._db.row_factory = aiosqlite.Row
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA busy_timeout=5000")
        await self._db.execute("PRAGMA foreign_keys=ON")
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                id TEXT PRIMARY KEY,
                user_id TEXT NOT NULL,
                title TEXT NOT NULL DEFAULT 'New chat',
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        await self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                sources TEXT NOT NULL DEFAULT '[]',
                created_at REAL NOT NULL,
                prompt_tokens INTEGER NOT NULL DEFAULT 0,
                completion_tokens INTEGER NOT NULL DEFAULT 0,
                cost REAL NOT NULL DEFAULT 0,
                latency_ms REAL NOT NULL DEFAULT 0
            )
            """
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_user_updated ON sessions(user_id, updated_at DESC)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at)"
        )
        # Migration for existing databases created before token/cost tracking.
        cols = await self._db.execute_fetchall("PRAGMA table_info(messages)")
        col_names = {row["name"] for row in cols}
        if "prompt_tokens" not in col_names:
            await self._db.execute("ALTER TABLE messages ADD COLUMN prompt_tokens INTEGER NOT NULL DEFAULT 0")
            await self._db.execute("ALTER TABLE messages ADD COLUMN completion_tokens INTEGER NOT NULL DEFAULT 0")
            await self._db.execute("ALTER TABLE messages ADD COLUMN cost REAL NOT NULL DEFAULT 0")
        if "latency_ms" not in col_names:
            await self._db.execute("ALTER TABLE messages ADD COLUMN latency_ms REAL NOT NULL DEFAULT 0")
        await self._db.commit()

    async def close(self) -> None:
        if self._db is not None:
            await self._db.close()
            self._db = None

    async def create_session(self, user_id: str, title: str = "New chat") -> SessionOut:
        session_id = uuid.uuid4().hex
        ts = _now()
        await self._db.execute(
            "INSERT INTO sessions (id, user_id, title, created_at, updated_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, user_id, title, ts, ts),
        )
        await self._db.commit()
        return SessionOut(id=session_id, title=title, created_at=ts, updated_at=ts)

    async def list_sessions(self, user_id: str, limit: int = 100) -> list[SessionOut]:
        rows = await self._db.execute_fetchall(
            """
            SELECT s.id, s.title, s.created_at, s.updated_at,
                   (SELECT m.content FROM messages m
                     WHERE m.session_id = s.id ORDER BY m.created_at DESC, m.id DESC LIMIT 1) AS last,
                   (SELECT COALESCE(SUM(m.cost), 0) FROM messages m WHERE m.session_id = s.id) AS total_cost
            FROM sessions s
            WHERE s.user_id = ?
            ORDER BY s.updated_at DESC
            LIMIT ?
            """,
            (user_id, limit),
        )
        out = []
        for r in rows:
            last = (r["last"] or "").strip()
            out.append(
                SessionOut(
                    id=r["id"],
                    title=r["title"],
                    created_at=r["created_at"],
                    updated_at=r["updated_at"],
                    last_preview=last[:PREVIEW_LEN],
                    total_cost=float(r["total_cost"] or 0.0),
                )
            )
        return out

    async def _fetchone(self, query: str, params: tuple = ()):
        rows = await self._db.execute_fetchall(query, params)
        return rows[0] if rows else None

    async def _fetchall(self, query: str, params: tuple = ()):
        return await self._db.execute_fetchall(query, params)

    async def get_session(self, session_id: str, user_id: str) -> SessionOut | None:
        row = await self._fetchone(
            "SELECT id, title, created_at, updated_at FROM sessions WHERE id = ? AND user_id = ?",
            (session_id, user_id),
        )
        if row is None:
            return None
        return SessionOut(
            id=row["id"], title=row["title"], created_at=row["created_at"], updated_at=row["updated_at"]
        )

    async def messages(self, session_id: str, user_id: str) -> list[MessageOut]:
        if await self.get_session(session_id, user_id) is None:
            raise HTTPException(status_code=404, detail="conversation not found")
        rows = await self._db.execute_fetchall(
            """
            SELECT id, role, content, sources, created_at, prompt_tokens, completion_tokens, cost, latency_ms
            FROM messages WHERE session_id = ? ORDER BY created_at ASC, id ASC
            """,
            (session_id,),
        )
        return [_row_to_message(r) for r in rows]

    async def append_message(
        self,
        session_id: str,
        user_id: str,
        role: str,
        content: str,
        sources: list[dict] | None = None,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost: float = 0.0,
        latency_ms: float = 0.0,
    ) -> MessageOut:
        if await self.get_session(session_id, user_id) is None:
            raise HTTPException(status_code=404, detail="conversation not found")
        ts = _now()
        cur = await self._db.execute(
            "INSERT INTO messages (session_id, role, content, sources, created_at, prompt_tokens, completion_tokens, cost, latency_ms)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (session_id, role, content, json_dumps(sources or []), ts, prompt_tokens, completion_tokens, cost, latency_ms),
        )
        await self._db.execute(
            "UPDATE sessions SET updated_at = ?, title = COALESCE(NULLIF(title, ''), 'New chat') WHERE id = ?",
            (ts, session_id),
        )
        await self._db.commit()
        return MessageOut(
            id=cur.lastrowid,
            role=role,
            content=content,
            sources=sources or [],
            created_at=ts,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
            latency_ms=latency_ms,
        )

    async def rename_session(self, session_id: str, user_id: str, title: str) -> SessionOut:
        session = await self.get_session(session_id, user_id)
        if session is None:
            raise HTTPException(status_code=404, detail="conversation not found")
        clean = (title or "").strip()[:200]
        await self._db.execute("UPDATE sessions SET title = ? WHERE id = ?", (clean, session_id))
        await self._db.commit()
        return SessionOut(
            id=session_id,
            title=clean,
            created_at=session.created_at,
            updated_at=session.updated_at,
        )

    async def delete_session(self, session_id: str, user_id: str) -> None:
        if await self.get_session(session_id, user_id) is None:
            raise HTTPException(status_code=404, detail="conversation not found")
        await self._db.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        await self._db.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        await self._db.commit()

    async def recent_turns(self, session_id: str, user_id: str, max_turns: int) -> list[MessageOut]:
        """The most recent `max_turns` user/assistant message pairs (oldest
        first), used as conversation context for the LLM prompt."""
        rows = await self._db.execute_fetchall(
            """
            SELECT id, role, content, sources, created_at, prompt_tokens, completion_tokens, cost, latency_ms
            FROM messages WHERE session_id = ? ORDER BY created_at DESC, id DESC LIMIT ?
            """,
            (session_id, max_turns * 2),
        )
        rows.reverse()
        return [_row_to_message(r) for r in rows]

    async def purge_expired(self) -> int:
        """Delete conversations idle for CHAT_RETENTION_DAYS or longer."""
        cutoff = _now() - config.CHAT_RETENTION_DAYS * 86400
        stale = await self._db.execute_fetchall("SELECT id FROM sessions WHERE updated_at < ?", (cutoff,))
        for r in stale:
            await self._db.execute("DELETE FROM messages WHERE session_id = ?", (r["id"],))
            await self._db.execute("DELETE FROM sessions WHERE id = ?", (r["id"],))
        await self._db.commit()
        return len(stale)

    async def stats(self, user_id: str) -> SessionStatsOut:
        """Aggregate token/cost usage across the user's conversations."""
        sessions_row = await self._fetchone(
            "SELECT COUNT(*) AS n FROM sessions WHERE user_id = ?", (user_id,)
        )
        msgs_row = await self._fetchone(
            """
            SELECT COUNT(*) AS n,
                   COALESCE(SUM(prompt_tokens), 0) AS pt,
                   COALESCE(SUM(completion_tokens), 0) AS ct,
                   COALESCE(SUM(cost), 0) AS cost
            FROM messages WHERE session_id IN (SELECT id FROM sessions WHERE user_id = ?)
            """,
            (user_id,),
        )
        return SessionStatsOut(
            sessions=int(sessions_row["n"]) if sessions_row else 0,
            messages=int(msgs_row["n"]) if msgs_row else 0,
            total_tokens=int((msgs_row["pt"] if msgs_row else 0) + (msgs_row["ct"] if msgs_row else 0)),
            total_cost=float(msgs_row["cost"] if msgs_row else 0.0),
        )

    async def global_stats(self) -> dict:
        """Cross-user analytics across the whole chat DB (privacy-safe: no
        message contents, only counts/aggregates). Never raises."""
        try:
            sessions_row = await self._fetchone(
                "SELECT COUNT(*) AS n FROM sessions"
            )
            users_row = await self._fetchone("SELECT COUNT(DISTINCT user_id) AS n FROM sessions")
            msgs_row = await self._fetchone(
                """
                SELECT COUNT(*) AS n,
                       COALESCE(SUM(CASE WHEN role='assistant' THEN prompt_tokens END), 0) AS pt,
                       COALESCE(SUM(CASE WHEN role='assistant' THEN completion_tokens END), 0) AS ct,
                       COALESCE(SUM(cost), 0) AS cost,
                       COALESCE(AVG(CASE WHEN role='assistant' AND latency_ms > 0 THEN latency_ms END), 0) AS latency
                FROM messages
                """
            )
            top_cost = await self._fetchall(
                """
                SELECT s.title, s.updated_at,
                       COUNT(m.id) AS messages,
                       COALESCE(SUM(m.cost), 0) AS cost
                FROM sessions s JOIN messages m ON m.session_id = s.id
                GROUP BY s.id ORDER BY cost DESC LIMIT 10
                """
            )
            top_messages = await self._fetchall(
                """
                SELECT s.title, s.updated_at,
                       COUNT(m.id) AS messages,
                       COALESCE(SUM(m.prompt_tokens + m.completion_tokens), 0) AS tokens
                FROM sessions s JOIN messages m ON m.session_id = s.id
                GROUP BY s.id ORDER BY tokens DESC LIMIT 10
                """
            )
            today = datetime.now(UTC).strftime("%Y-%m-%d")
            day_rows = await self._fetchall(
                """
                SELECT date(created_at, 'unixepoch') AS d, COUNT(*) AS n
                FROM sessions GROUP BY d ORDER BY d DESC LIMIT 14
                """
            )
            return {
                "sessions": int(sessions_row["n"]) if sessions_row else 0,
                "users": int(users_row["n"]) if users_row else 0,
                "messages": int(msgs_row["n"]) if msgs_row else 0,
                "total_tokens": int((msgs_row["pt"] if msgs_row else 0) + (msgs_row["ct"] if msgs_row else 0)),
                "total_cost": float(msgs_row["cost"] if msgs_row else 0.0),
                "avg_latency_ms": round(float(msgs_row["latency"] if msgs_row else 0.0), 1),
                "top_by_cost": [[r["title"], int(r["messages"]), round(float(r["cost"]), 4), r["updated_at"]] for r in top_cost],
                "top_by_tokens": [[r["title"], int(r["messages"]), int(r["tokens"]), r["updated_at"]] for r in top_messages],
                "sessions_today": sum(int(r["n"]) for r in day_rows if r["d"] == today),
                "daily_sessions": [[r["d"], int(r["n"])] for r in day_rows],
            }
        except Exception:
            logger.exception("chat global_stats failed")
            return {"error": "chat analytics unavailable"}


def json_dumps(v) -> str:
    return json.dumps(v, separators=(",", ":"))


def json_loads(s: str) -> list[dict]:
    try:
        return json.loads(s or "[]")
    except (ValueError, TypeError):
        return []


def _row_to_message(r) -> MessageOut:
    return MessageOut(
        id=r["id"],
        role=r["role"],
        content=r["content"],
        sources=json_loads(r["sources"]),
        created_at=r["created_at"],
        prompt_tokens=int(r["prompt_tokens"] or 0),
        completion_tokens=int(r["completion_tokens"] or 0),
        cost=float(r["cost"] or 0.0),
        latency_ms=float(r["latency_ms"] or 0.0),
    )


_SMALLTALK_PATTERNS: dict[str, str] = {
    r"^(hi|hii+|hey|hello|yo|hola|howdy|namaste|good (morning|afternoon|evening))\b": (
        "Hello! I'm ASK VCCircle. Ask me about VCCircle's business news archive — "
        "deals, funding, IPOs, M&A, companies, or a specific sector."
    ),
    r"^(thanks|thank you|ty|thx|thank u|cheers)\b": "You're welcome! Ask me anything about VCCircle's news archive anytime.",
    r"^(goodbye|bye|see you|gtg|cya)\b": "Goodbye! Come back anytime to search VCCircle's archive.",
    r"\bhow are you\b": "I'm doing great, thanks for asking! What would you like to know about VCCircle's news archive?",
    r"\bwho are you\b": "I'm ASK VCCircle, an AI assistant that searches VCCircle's business news archive. Ask me about deals, funding, IPOs, companies, or sectors.",
    r"\bwhat can you do\b|how (do|can) you work|what are you": "I search VCCircle's archive for relevant articles and summarize answers with citations. Try asking, e.g., \"top 10 fintech deals 2025\" or \"who invested in Ola Electric?\".",
    r"\b(can|are) you help( me)?\b": "Of course! Ask me anything about VCCircle's business news archive — deals, funding, IPOs, companies, or sectors.",
}


def _smalltalk_reply(question: str) -> str | None:
    """Return a canned friendly reply for greetings/thanks/small talk, or None
    when the message looks like a real query for the archive."""
    q = question.strip().lower()
    if not q or len(q.split()) > 12:
        return None
    for pattern, reply in _SMALLTALK_PATTERNS.items():
        if re.search(pattern, q):
            return reply
    return None


@dataclass
class PreparedTurn:
    """Outcome of retrieval + prompt building for one turn.

    Either carries a ready-made `answer` (small talk, no sources, or weak
    results) with zero token usage, or a `prompt` for the LLM plus the sources
    to cite. `needs_llm` distinguishes the two.
    """

    answer: str
    sources: list[dict]
    note: str | None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    needs_llm: bool = False


async def _prepare_turn(question: str, history: list[MessageOut]) -> PreparedTurn:
    smalltalk = _smalltalk_reply(question)
    if smalltalk is not None:
        return PreparedTurn(answer=smalltalk, sources=[], note=None)

    from app.answer_fallback import date_label, fallback_answer, results_are_weak, weak_results_note
    from app.main import _effective_intent, retrieve_and_rerank, source_context, to_summary

    retrieval_q, eff_from, eff_to = _effective_intent(question, None, None)
    reranked = await retrieve_and_rerank(retrieval_q, config.TOP_K, None)
    sources = [s for s in reranked if s.score >= config.ASK_MIN_SCORE][: config.TOP_K]

    note = (
        weak_results_note([s.score for s in sources], date_label(eff_from, eff_to))
        if config.ENABLE_WEAK_FALLBACK
        else None
    )

    if not sources:
        return PreparedTurn(answer="No sufficiently relevant articles were found for this query.", sources=[], note=note)

    if config.ENABLE_WEAK_FALLBACK and results_are_weak([s.score for s in sources]):
        return PreparedTurn(
            answer=fallback_answer(question, len(sources), date_label(eff_from, eff_to)),
            sources=[to_summary(s).model_dump() for s in sources],
            note=note,
        )

    context = "\n\n".join(source_context(s, i + 1) for i, s in enumerate(sources))
    history_text = "\n".join(f"{m.role}: {m.content}" for m in history if m.role in ("user", "assistant"))
    prompt = CHAT_PROMPT.format(history=history_text or "(none)", context=context, question=question)
    return PreparedTurn(
        answer=prompt,
        sources=[to_summary(s).model_dump() for s in sources],
        note=note,
        needs_llm=True,
    )


async def _run_turn(question: str, history: list[MessageOut]) -> tuple[str, list[dict], str | None, int, int, float]:
    """Retrieve, build a conversation-aware prompt, and call the LLM.

    Returns (answer, sources, note, prompt_tokens, completion_tokens, cost).
    Uses the shared retrieval pipeline from app.main; imported lazily to avoid a
    circular import with app.main. Raises BudgetExceeded when the daily LLM
    spend cap is already exhausted."""
    turn = await _prepare_turn(question, history)
    if not turn.needs_llm:
        return turn.answer, turn.sources, turn.note, turn.prompt_tokens, turn.completion_tokens, turn.cost

    await assert_within_budget()
    result = await generate_answer(state_llm(), turn.answer, config.LLM_MODEL)
    await record_cost(result.cost())
    return (
        result.content,
        turn.sources,
        turn.note,
        result.prompt_tokens,
        result.completion_tokens,
        result.cost(),
    )


def state_llm():
    from app import main

    return main.state.get("llm")


CHAT_PROMPT = """You are ASK VCCircle, a helpful assistant that answers questions about VCCircle's \
business news archive using ONLY the numbered articles below. Cite the article number(s) for every \
factual claim, like [1] or [2][3]. If the user asks a follow-up question, use the conversation history \
for context but only make claims supported by the articles. If the articles contain no relevant \
information, say so plainly instead of guessing.

Conversation so far:
{history}

Articles:
{context}

Question: {question}

Answer (with inline [n] citations):"""


def _require_store() -> ChatStore:
    if store is None:
        raise HTTPException(status_code=503, detail="chat store not initialized")
    return store


def _validate_question(body: MessageIn) -> str:
    question = (body.content or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="empty message")
    if len(question) > MAX_CONTENT_LEN:
        raise HTTPException(status_code=400, detail=f"message too long (max {MAX_CONTENT_LEN} chars)")
    return question


async def _start_turn(s: ChatStore, session_id: str, user_id: str, question: str) -> tuple[MessageOut, list[MessageOut]]:
    user_msg = await s.append_message(session_id, user_id, "user", question)
    history = await s.recent_turns(session_id, user_id, config.CHAT_MAX_HISTORY_TURNS)
    return user_msg, history


async def _auto_title(s: ChatStore, session_id: str, user_id: str, question: str) -> None:
    session = await s.get_session(session_id, user_id)
    if session is not None and session.title.strip() in ("", "New chat"):
        await s.rename_session(session_id, user_id, question[:60] or "New chat")


@router.post("/sessions", response_model=SessionOut)
async def create_session(x_user_id: str | None = Header(default=None, alias="X-User-Id")):
    user_id = _user_id(x_user_id)
    return await _require_store().create_session(user_id)


@router.get("/sessions", response_model=list[SessionOut])
async def list_sessions(x_user_id: str | None = Header(default=None, alias="X-User-Id")):
    user_id = _user_id(x_user_id)
    return await _require_store().list_sessions(user_id)


@router.get("/sessions/{session_id}", response_model=SessionDetailOut)
async def get_session(session_id: str, x_user_id: str | None = Header(default=None, alias="X-User-Id")):
    user_id = _user_id(x_user_id)
    s = _require_store()
    session = await s.get_session(session_id, user_id)
    if session is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    messages = await s.messages(session_id, user_id)
    return SessionDetailOut(**session.model_dump(), messages=messages)


@router.patch("/sessions/{session_id}", response_model=SessionOut)
async def rename_session(session_id: str, body: MessageIn, x_user_id: str | None = Header(default=None, alias="X-User-Id")):
    user_id = _user_id(x_user_id)
    return await _require_store().rename_session(session_id, user_id, body.content)


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, x_user_id: str | None = Header(default=None, alias="X-User-Id")):
    user_id = _user_id(x_user_id)
    await _require_store().delete_session(session_id, user_id)
    return {"ok": True}


@router.get("/usage", response_model=SessionStatsOut)
async def get_usage(x_user_id: str | None = Header(default=None, alias="X-User-Id")):
    """Aggregate token usage and cost across the user's conversations."""
    user_id = _user_id(x_user_id)
    return await _require_store().stats(user_id)


@router.post("/sessions/{session_id}/messages", response_model=TurnOut)
async def send_message(
    session_id: str,
    body: MessageIn,
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
):
    user_id = _user_id(x_user_id)
    s = _require_store()
    question = _validate_question(body)

    user_msg, history = await _start_turn(s, session_id, user_id, question)
    start = time.perf_counter()

    try:
        answer, sources, note, prompt_tokens, completion_tokens, cost = await _run_turn(question, history)
    except BudgetExceeded:
        raise HTTPException(
            status_code=429,
            detail={"error": "Daily AI budget reached", "detail": "The daily chat budget is exhausted; please try again tomorrow."},
        )
    except LLMUnavailableError:
        raise HTTPException(
            status_code=503,
            detail={"error": "LLM temporarily unavailable", "detail": "The language model could not be reached; please retry shortly."},
        )

    latency_ms = (time.perf_counter() - start) * 1000

    assistant_msg = await s.append_message(
        session_id,
        user_id,
        "assistant",
        answer,
        sources,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost=cost,
        latency_ms=latency_ms,
    )
    await _auto_title(s, session_id, user_id, question)
    return TurnOut(user=user_msg, assistant=assistant_msg, note=note, latency_ms=latency_ms)


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, separators=(',', ':'))}\n\n"


@router.post("/sessions/{session_id}/messages/stream")
async def send_message_stream(
    session_id: str,
    body: MessageIn,
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
):
    """SSE-streamed chat turn: retrieves, then streams the LLM answer token by
    token. Events: 'start', 'delta' (content chunk), 'done' (with full message,
    sources, usage, cost, latency), or 'error'. The assistant message is saved
    once streaming completes."""
    user_id = _user_id(x_user_id)
    s = _require_store()
    question = _validate_question(body)

    user_msg, history = await _start_turn(s, session_id, user_id, question)

    async def event_stream():
        start = time.perf_counter()
        try:
            yield _sse("start", {"user": user_msg.model_dump()})
            turn = await _prepare_turn(question, history)
            if not turn.needs_llm:
                latency_ms = (time.perf_counter() - start) * 1000
                assistant_msg = await s.append_message(
                    session_id, user_id, "assistant", turn.answer, turn.sources,
                    prompt_tokens=turn.prompt_tokens, completion_tokens=turn.completion_tokens,
                    cost=turn.cost, latency_ms=latency_ms,
                )
                await _auto_title(s, session_id, user_id, question)
                yield _sse("done", {"message": assistant_msg.model_dump(), "note": turn.note, "latency_ms": latency_ms})
                return

            await assert_within_budget()
            usage_holder: list = []
            chunks: list[str] = []
            async for piece in stream_answer(state_llm(), turn.answer, config.LLM_MODEL, usage_holder):
                chunks.append(piece)
                yield _sse("delta", {"text": piece})

            usage = usage_holder[0] if usage_holder else None
            latency_ms = (time.perf_counter() - start) * 1000
            answer = "".join(chunks)
            result = LLMResult(
                content=answer,
                prompt_tokens=usage.prompt_tokens if usage else 0,
                completion_tokens=usage.completion_tokens if usage else 0,
            )
            await record_cost(result.cost())
            assistant_msg = await s.append_message(
                session_id, user_id, "assistant", answer, turn.sources,
                prompt_tokens=result.prompt_tokens,
                completion_tokens=result.completion_tokens,
                cost=result.cost(),
                latency_ms=latency_ms,
            )
            await _auto_title(s, session_id, user_id, question)
            yield _sse(
                "done",
                {
                    "message": assistant_msg.model_dump(),
                    "note": turn.note,
                    "latency_ms": latency_ms,
                },
            )
        except LLMUnavailableError:
            yield _sse("error", {"error": "LLM temporarily unavailable"})
        except BudgetExceeded:
            yield _sse("error", {"error": "Daily AI budget reached"})
        except Exception:
            logger.exception("chat stream turn failed")
            yield _sse("error", {"error": "Something went wrong"})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


async def retention_loop() -> None:
    """Background task: purge conversations idle past retention. Never raises."""
    while True:
        try:
            if store is not None:
                n = await store.purge_expired()
                if n:
                    logger.info("chat retention: purged %d expired conversation(s)", n)
        except Exception:
            logger.exception("chat retention purge failed")
        await asyncio.sleep(config.CHAT_PURGE_INTERVAL_SECONDS)