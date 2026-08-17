"""Per-user chat conversations stored in SQLite on the host.

Conversations survive restarts (unlike Redis-without-AOF) and are purged after
CHAT_RETENTION_DAYS of inactivity. Users are identified by an anonymous
X-User-Id header set by the frontend (device UUID); this is NOT authentication.

Turn flow (POST /api/chat/sessions/{id}/messages) reuses the same retrieval,
rerank, fallback and LLM pipeline as /ask, but builds a conversation-aware
prompt so the model can follow up on prior turns in the session.
"""

import asyncio
import logging
import os
import time
import uuid

import aiosqlite
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from app.config import config
from app.llm import LLMUnavailableError, generate_answer

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


class MessageOut(BaseModel):
    id: int
    role: str
    content: str
    sources: list[dict] = []
    created_at: float


class SessionDetailOut(SessionOut):
    messages: list[MessageOut] = []


class MessageIn(BaseModel):
    content: str


class TurnOut(BaseModel):
    user: MessageOut
    assistant: MessageOut
    note: str | None = None


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
                created_at REAL NOT NULL
            )
            """
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_user_updated ON sessions(user_id, updated_at DESC)"
        )
        await self._db.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, created_at)"
        )
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
                     WHERE m.session_id = s.id ORDER BY m.created_at DESC, m.id DESC LIMIT 1) AS last
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
                )
            )
        return out

    async def _fetchone(self, query: str, params: tuple = ()):
        rows = await self._db.execute_fetchall(query, params)
        return rows[0] if rows else None

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
            SELECT id, role, content, sources, created_at FROM messages
            WHERE session_id = ? ORDER BY created_at ASC, id ASC
            """,
            (session_id,),
        )
        return [
            MessageOut(
                id=r["id"],
                role=r["role"],
                content=r["content"],
                sources=json_loads(r["sources"]),
                created_at=r["created_at"],
            )
            for r in rows
        ]

    async def append_message(
        self, session_id: str, user_id: str, role: str, content: str, sources: list[dict] | None = None
    ) -> MessageOut:
        if await self.get_session(session_id, user_id) is None:
            raise HTTPException(status_code=404, detail="conversation not found")
        ts = _now()
        cur = await self._db.execute(
            "INSERT INTO messages (session_id, role, content, sources, created_at) VALUES (?, ?, ?, ?, ?)",
            (session_id, role, content, json_dumps(sources or []), ts),
        )
        await self._db.execute(
            "UPDATE sessions SET updated_at = ?, title = COALESCE(NULLIF(title, ''), 'New chat') WHERE id = ?",
            (ts, session_id),
        )
        await self._db.commit()
        return MessageOut(id=cur.lastrowid, role=role, content=content, sources=sources or [], created_at=ts)

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
            SELECT id, role, content, sources, created_at FROM messages
            WHERE session_id = ? ORDER BY created_at DESC, id DESC LIMIT ?
            """,
            (session_id, max_turns * 2),
        )
        rows.reverse()
        return [
            MessageOut(
                id=r["id"],
                role=r["role"],
                content=r["content"],
                sources=json_loads(r["sources"]),
                created_at=r["created_at"],
            )
            for r in rows
        ]

    async def purge_expired(self) -> int:
        """Delete conversations idle for CHAT_RETENTION_DAYS or longer."""
        cutoff = _now() - config.CHAT_RETENTION_DAYS * 86400
        stale = await self._db.execute_fetchall("SELECT id FROM sessions WHERE updated_at < ?", (cutoff,))
        for r in stale:
            await self._db.execute("DELETE FROM messages WHERE session_id = ?", (r["id"],))
            await self._db.execute("DELETE FROM sessions WHERE id = ?", (r["id"],))
        await self._db.commit()
        return len(stale)


def json_dumps(v) -> str:
    import json

    return json.dumps(v, separators=(",", ":"))


def json_loads(s: str) -> list[dict]:
    import json

    try:
        return json.loads(s or "[]")
    except (ValueError, TypeError):
        return []


async def _run_turn(question: str, history: list[MessageOut]) -> tuple[str, list[dict], str | None]:
    """Retrieve, build a conversation-aware prompt, and call the LLM.

    Returns (answer, sources, note). Mirrors /ask's pipeline; imported lazily to
    avoid a circular import with app.main."""
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
        return "No sufficiently relevant articles were found for this query.", [], note

    if config.ENABLE_WEAK_FALLBACK and results_are_weak([s.score for s in sources]):
        return fallback_answer(question, len(sources), date_label(eff_from, eff_to)), [to_summary(s).model_dump() for s in sources], note

    context = "\n\n".join(source_context(s, i + 1) for i, s in enumerate(sources))
    history_text = "\n".join(f"{m.role}: {m.content}" for m in history if m.role in ("user", "assistant"))
    prompt = CHAT_PROMPT.format(history=history_text or "(none)", context=context, question=question)

    answer = await generate_answer(state_llm(), prompt, config.LLM_MODEL)
    return answer, [to_summary(s).model_dump() for s in sources], note


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


@router.post("/sessions/{session_id}/messages", response_model=TurnOut)
async def send_message(
    session_id: str,
    body: MessageIn,
    x_user_id: str | None = Header(default=None, alias="X-User-Id"),
):
    user_id = _user_id(x_user_id)
    s = _require_store()
    question = (body.content or "").strip()
    if not question:
        raise HTTPException(status_code=400, detail="empty message")
    if len(question) > MAX_CONTENT_LEN:
        raise HTTPException(status_code=400, detail=f"message too long (max {MAX_CONTENT_LEN} chars)")

    user_msg = await s.append_message(session_id, user_id, "user", question)
    history = await s.recent_turns(session_id, user_id, config.CHAT_MAX_HISTORY_TURNS)

    try:
        answer, sources, note = await _run_turn(question, history)
    except LLMUnavailableError:
        raise HTTPException(
            status_code=503,
            detail={"error": "LLM temporarily unavailable", "detail": "The language model could not be reached; please retry shortly."},
        )

    assistant_msg = await s.append_message(session_id, user_id, "assistant", answer, sources)
    session = await s.get_session(session_id, user_id)
    if session is not None and session.title.strip() in ("", "New chat"):
        first_user = question[:60] or "New chat"
        await s.rename_session(session_id, user_id, first_user)
    return TurnOut(user=user_msg, assistant=assistant_msg, note=note)


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