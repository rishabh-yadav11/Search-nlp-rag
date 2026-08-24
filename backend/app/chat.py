"""Per-user chat conversations stored in SQLite.

Conversations survive restarts and are purged after CHAT_RETENTION_DAYS of
inactivity. The caller must be authenticated (Bearer token, validated by the
auth dependency at the router level); conversations are scoped to the
authenticated account's user id.

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
from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.auth import require_auth, require_permission
from app.config import config
from app.cost_budget import BudgetExceeded, assert_within_budget, record_cost
from app.llm import LLMResult, LLMUnavailableError, generate_answer, stream_answer
from app.query_intent import extract_year_range, suggested_top_k

logger = logging.getLogger("chat")

router = APIRouter(
    prefix="/api/chat",
    tags=["chat"],
    dependencies=[Depends(require_auth), Depends(require_permission("chat:use"))],
)

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


# Words that carry no retrieval topic on their own: pronouns, chart/table
# request verbs, output-format nouns, and bare follow-up fillers. A question
# made only of these (e.g. 'make this into a table', 'plot it', 'more') is a
# vague follow-up: it must inherit the previous turn's topic + date filter to
# retrieve anything meaningful.
_VAGUE_WORDS = frozenset((
    "a", "an", "the", "this", "that", "it", "them", "these", "those", "they",
    "there", "above", "below", "same", "such", "one", "some", "like", "as",
    "into", "in", "on", "for", "of", "and", "or", "with", "to", "please",
    "now", "again", "also", "then", "next", "about", "further", "more",
    "elaborate", "explain", "detail", "details", "why", "what", "how",
    "make", "create", "show", "draw", "give", "build", "plot", "display",
    "present", "convert", "format", "chart", "table", "graph", "pie", "bar",
    "line", "column", "area", "pictogram", "pictograph", "diagram", "visual",
    "visualize", "visualise", "visualization", "visualisation",
))


def _is_vague_followup(question: str) -> bool:
    """True when ``question`` has no standalone retrieval topic — a pure
    format/pronoun follow-up like 'make this into a table' or 'plot it' that
    must inherit the previous turn's topic+date filter to retrieve anything."""
    from app.query_intent import _strip_noise_words, range_query_topic

    topic = range_query_topic(question) or _strip_noise_words(question) or ""
    words = set(re.findall(r"[a-z]+", topic.lower()))
    meaningful = {w for w in words if w not in _VAGUE_WORDS and len(w) > 1}
    return not meaningful


def _previous_user_question(history: list[MessageOut]) -> str | None:
    """The user's question from the turn before the current one, or None. The
    history's last message is the just-appended current question."""
    seen_current = False
    for m in reversed(history):
        if m.role == "user":
            if seen_current:
                return m.content
            seen_current = True
    return None


_DATAVIZ_FENCE_RE = re.compile(r"```dataviz\s*\n(.*?)\n```", re.DOTALL)


def _as_float(v: object) -> float | None:
    """Coerce a cell to float (ints, floats, or digit strings), else None."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.replace(",", ""))
        except ValueError:
            return None
    return None


_MISSING_VALUE_TOKENS = frozenset((
    "", "value not stated", "not stated", "n/a", "na", "n/d", "nil", "none",
    "unknown", "tbd", "to be decided", "to be determined", "—", "-", "--",
))


def _missing_cell(v: object) -> bool:
    """True when a dataviz value cell marks a missing value (null, empty string,
    or a common 'not stated' token), so a top-N table can include items whose
    value isn't stated."""
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip().lower() in _MISSING_VALUE_TOKENS
    return False


def _valid_value_column(rows: list[list[object]], j: int) -> bool:
    """A value column must hold a number in every non-missing cell and at least
    one number overall (empty cells are allowed, e.g. 'value not stated')."""
    present = [r[j] for r in rows if not _missing_cell(r[j])]
    return bool(present) and all(_as_float(v) is not None for v in present)


def _first_numeric_column(rows: list[list[object]]) -> int | None:
    if not rows or not rows[0]:
        return None
    for j in range(len(rows[0])):
        if _valid_value_column(rows, j):
            return j
    return None


def _has_label_content(rows: list[list[object]], columns: list[str], value_column: int | None) -> bool:
    """A dataviz block is only useful if at least one non-value column carries
    identifying text. A 'top deals' table whose Deal cells are all empty shows
    only numbers and is treated as malformed so the nudge retry rebuilds it."""
    label_cols = [j for j in range(len(columns)) if j != value_column]
    if not label_cols:
        return True
    return any(not _missing_cell(r[j]) for j in label_cols for r in rows)


def parse_dataviz(text: str) -> dict | None:
    """Extract and validate the assistant's ``dataviz`` JSON data block.

    Returns the parsed block dict, or None when absent or malformed. The block
    powers the frontend table/bar/pie renderer; the prose answer always stands
    alone, so an invalid block is simply dropped instead of breaking chat. A
    block whose label cells are all empty (e.g. every Deal name blank) is
    malformed too: it conveys no information to the user."""
    m = _DATAVIZ_FENCE_RE.search(text or "")
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    columns = data.get("columns")
    rows = data.get("rows")
    if not isinstance(columns, list) or not columns or not all(isinstance(c, str) for c in columns):
        return None
    if not isinstance(rows, list) or not rows or not all(isinstance(r, list) for r in rows):
        return None
    if any(len(r) != len(columns) for r in rows):
        return None
    vc = data.get("value_column")
    if not isinstance(vc, int) or isinstance(vc, bool) or not (0 <= vc < len(columns)):
        vc = _first_numeric_column(rows)
    # A table (view='table' or explicit table ask) may have no numeric column at
    # all (e.g. every item's value is 'not stated'): value_column stays None and
    # the UI renders a plain text table. Chart views (bar/line/pie/picto) and
    # generic blocks without a table view still require a numeric column.
    if vc is not None and not _valid_value_column(rows, vc):
        return None
    if vc is None and data.get("view") != "table":
        return None
    if not _has_label_content(rows, columns, vc):
        return None
    data["value_column"] = vc
    return data


def _sanitize_dataviz(text: str) -> str:
    """Return ``text`` with any malformed ``dataviz`` fence removed.

    Valid blocks pass through untouched (the frontend renders them); malformed
    JSON is stripped so users never see raw, unparseable blocks."""
    if not text or "```dataviz" not in text:
        return text

    def _keep(match: re.Match) -> str:
        return match.group(0) if parse_dataviz(match.group(0)) is not None else ""

    return _DATAVIZ_FENCE_RE.sub(_keep, text)


def _finalize_answer(text: str, question: str) -> str:
    """Clean the raw LLM answer for storage/display.

    Charts are ONLY shown on an explicit request: if the user did not ask for a
    chart/graph/plot/table, any dataviz block the model emitted anyway is
    removed (guarding against non-deterministic emission). When a chart IS
    requested, the block is pinned to the requested view and malformed fences
    are stripped."""
    if _CHART_INTENT_RE.search(question):
        return _sanitize_dataviz(_apply_requested_view(text, question))
    if not text or "```dataviz" not in text:
        return text
    return _DATAVIZ_FENCE_RE.sub("", text).rstrip()


def _effective_chat_k(question: str) -> int:
    """How many sources chat should retrieve/cite for a question.

    Mirrors /search: a 'top N' request (or a bare top/best list) scales the
    source count up, floored at TOP_K and capped at CHAT_MAX_SOURCES so the LLM
    context stays bounded."""
    return min(max(config.TOP_K, suggested_top_k(question) or 0), config.CHAT_MAX_SOURCES)


# Matches only an EXPLICIT request for a chart/graph/plot/visualization/table,
# so ranked-list and numeric-comparison questions that do NOT mention a visual
# view get plain prose instead of an automatic chart (see CHAT_PROMPT).
_CHART_INTENT_RE = re.compile(
    r"\b(charts?|graphs?|pictogram|pictograph|diagram|visuali[sz]e|visuali[sz]ation|visual)\b"
    r"|(?:show|draw|make|create|give|build|plot)\s+(?:me\s+)?(?:a\s+|the\s+)?"
    r"(?:bar|line|pie|column|area)?\s*(?:chart|graph|plot|table)\b"
    r"|\b(?:as|in|into)\s+a\s+(?:chart|graph|plot|table)\b"
    r"|\b(?:chart|graph|plot)\s+(?:it|this|these|them|that|out)\b",
    re.IGNORECASE,
)

# Replaced by _dataviz_nudge() so the retry's row cap matches the requested N.
_DATAVIZ_MAX_ROWS_TOKEN = "{MAX_ROWS}"

_DATAVIZ_NUDGE = (
    "\n\nYour previous answer did not include a VALID JSON data block. You were asked to show a chart, "
    "graph, plot, or table, so re-answer the SAME question and END your answer with exactly one "
    "fenced code block tagged dataviz containing ONLY valid JSON, like this:\n\n"
    "```dataviz\n"
    '{"title": "Top deals", "columns": ["Deal", "Value ($B)"], "rows": [["Zepto", 1.0], ["Shriram Finance stake", 4.4]], "value_column": 1, "format": "$B"}\n'
    "```\n\n"
    "Rules: valid JSON only (double-quoted keys, no trailing commas, no markdown bullet lists); rows are "
    f"the ranked items (max {_DATAVIZ_MAX_ROWS_TOKEN} rows); value_column is the integer index of the "
    "numeric column and every cell in that column is a plain number actually stated in the articles; "
    "the first (label) column must contain every item's name — never leave it empty; "
    "never invent numbers; keep [n] citations only in the prose."
)


def _dataviz_nudge(question: str) -> str:
    """The dataviz retry instruction with the row cap set to the question's
    effective source count, so a 'top 10' chart can actually hold 10 rows."""
    nudge = _DATAVIZ_NUDGE.replace(_DATAVIZ_MAX_ROWS_TOKEN, str(_effective_chat_k(question)))
    view = _requested_view(question)
    if view is not None:
        nudge += f" The user explicitly asked for a {view} view — emit that exact type of data block."
    return nudge


# Canonical dataviz views exposed by the frontend (DataViz.tsx), in match
# priority (most specific first; 'graph' is the generic bar fallback).
_VIEW_TERMS: list[tuple[str, str]] = [
    ("pictogram", "picto"),
    ("pictograph", "picto"),
    ("line", "line"),
    ("pie", "pie"),
    ("donut", "pie"),
    ("bar", "bar"),
    ("column", "bar"),
    ("histogram", "bar"),
    ("table", "table"),
    ("tabular", "table"),
    ("graph", "bar"),
]


def _requested_view(question: str) -> str | None:
    """The canonical dataviz view the user explicitly asked for
    (table/bar/line/pie/picto), or None for a generic chart request."""
    if _CHART_INTENT_RE.search(question) is None:
        return None
    q = question.lower()
    for term, view in _VIEW_TERMS:
        if re.search(rf"\b{term}\b", q):
            return view
    return None


def _dataviz_view_instruction(question: str) -> str:
    """Prompt sentence pinning the data block to the explicitly requested view,
    or '' when the user only asked for a generic chart."""
    view = _requested_view(question)
    if view is None:
        return ""
    return (
        f"The user explicitly asked for a {view} view. Structure the data block for that exact view: "
        f"bar/line/pie need a numeric value column, pictogram values must be non-negative integers, and "
        f'a table can hold any columns. Set "kind" to "{view}" when it is bar, line, or pie.'
    )


def _parse_dataviz_with_view(text: str, view: str) -> dict | None:
    """Like parse_dataviz but with ``view`` pre-applied, so a value-less block
    (value_column null) is accepted for an explicit table ask."""
    m = _DATAVIZ_FENCE_RE.search(text or "")
    if not m:
        return None
    try:
        data = json.loads(m.group(1))
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    data["view"] = view
    return parse_dataviz("```dataviz\n" + json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n```")


def _apply_requested_view(text: str, question: str) -> str:
    """Pin the dataviz block's ``view`` to the visualization the user explicitly
    asked for, so the frontend renders ONLY that view (no view toggles). No-op
    for generic chart asks or answers without a valid block."""
    view = _requested_view(question)
    if view is None or not text or "```dataviz" not in text:
        return text

    def _rewrite(match: re.Match) -> str:
        block = _parse_dataviz_with_view(match.group(0), view)
        if block is None:
            return match.group(0)
        if view in ("bar", "line", "pie"):
            block["kind"] = view
        return "```dataviz\n" + json.dumps(block, ensure_ascii=False, separators=(",", ":")) + "\n```"

    return _DATAVIZ_FENCE_RE.sub(_rewrite, text)


async def _answer_with_dataviz(question: str, prompt: str) -> LLMResult:
    """Call the LLM once, nudging it to include a dataviz data block when the
    question explicitly asks for a chart/graph/plot/table and the model skipped
    the block. One extra call at most; token usage is summed. A failed nudge
    retry keeps the first answer instead of erroring the turn."""
    result = await generate_answer(state_llm(), prompt, config.LLM_MODEL)
    if parse_dataviz(result.content) is None and _CHART_INTENT_RE.search(question):
        try:
            nudge = await generate_answer(state_llm(), prompt + _dataviz_nudge(question), config.LLM_MODEL)
        except LLMUnavailableError:
            return result
        result.content = nudge.content
        result.prompt_tokens += nudge.prompt_tokens
        result.completion_tokens += nudge.completion_tokens
    return result


# Refusal signatures a model can emit instead of the requested ranked list —
# 'cannot be generated', 'unable to provide', 'do not contain specific amounts'.
_RANKING_REFUSAL_RE = re.compile(
    r"\b(?:cannot|can'?t|unable|couldn'?t|won'?t)\s+(?:be\s+)?"
    r"(?:generated|provided|ranked|determined|constructed|compiled|created|listed)\b"
    r"|\b(?:cannot|can'?t|unable\s+to)\s+(?:generate|provide|rank|determine|construct|compile)\b"
    r"|\b(?:do|does)\s+not\s+(?:contain|include|have|provide)\s+(?:specific\s+)?"
    r"(?:offering\s+)?(?:amounts?|values?|figures?|data|proceeds)\b"
    r"|\bno\s+(?:specific\s+)?(?:offering\s+)?(?:amounts?|values?|figures?|data|proceeds)"
    r"\s+(?:available|stated|provided|exist)\b",
    re.IGNORECASE,
)


def _is_ranking_refusal(text: str) -> bool:
    """True when an answer refuses to produce a ranked list because values are
    missing instead of ranking the named items (value not stated)."""
    return bool(text and _RANKING_REFUSAL_RE.search(text))


def _is_ranking_question(question: str) -> bool:
    """True for a ranked/numeric list question (a 'top N' or top/best/leading/
    biggest/largest intent), for which a refusal must trigger a nudge retry."""
    return suggested_top_k(question) is not None


_RANKING_NUDGE = (
    "\n\nYour previous answer refused to provide a ranked list because exact values were missing. "
    "Re-answer the SAME question and ALWAYS produce the ranked list. Use only items the articles name; "
    "order them by whatever is known (value, size, prominence, or recency); and write \"value not "
    "stated\" for every item whose amount is not in the articles. Never claim the list cannot be "
    "generated or ranked — a ranked list with \"value not stated\" entries is always better than a refusal."
)


async def _answer_ranked(question: str, prompt: str) -> LLMResult:
    """Call the LLM for a chat answer, applying the dataviz nudge (when a chart
    was asked) and the ranking-refusal nudge (when a ranked list came back as a
    refusal). At most one extra call for each; a failed retry keeps the first
    answer instead of erroring the turn."""
    result = await _answer_with_dataviz(question, prompt)
    if _is_ranking_question(question) and _is_ranking_refusal(result.content):
        try:
            nudge = await generate_answer(state_llm(), prompt + _RANKING_NUDGE, config.LLM_MODEL)
        except LLMUnavailableError:
            return result
        result.content = nudge.content
        result.prompt_tokens += nudge.prompt_tokens
        result.completion_tokens += nudge.completion_tokens
    return result


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
    from app.main import (
        _effective_intent,
        body_rescue,
        build_facet_filter,
        retrieve_and_rerank,
        source_context,
        to_summary,
    )

    retrieval_q, eff_from, eff_to, dealtype, industry = _effective_intent(question, None, None)
    k = _effective_chat_k(question)
    prev_question = _previous_user_question(history)
    if prev_question and _is_vague_followup(question):
        # A vague follow-up ('make this into a table', 'plot it', 'more') has no
        # standalone topic to retrieve on — 'make this into a table' as an
        # embedding query finds nothing and short-circuits the turn. Inherit the
        # previous turn's retrieval query, date filter, and category facets (and
        # top-N size) so the same sources are re-presented, while the LLM still
        # gets full history.
        prev_q, prev_from, prev_to, prev_dealtype, prev_industry = _effective_intent(prev_question, None, None)
        rng = extract_year_range(question)
        if rng:
            from app.query_intent import extract_list_topic, range_query_topic

            retrieval_q = range_query_topic(prev_question) or extract_list_topic(prev_question) or prev_q
            eff_from, eff_to = rng[0], rng[1]
        else:
            retrieval_q, eff_from, eff_to = prev_q, prev_from, prev_to
        dealtype, industry = prev_dealtype, prev_industry
        k = _effective_chat_k(prev_question)
    qfilter = build_facet_filter(industry, dealtype, None, eff_from, eff_to)
    reranked = await retrieve_and_rerank(retrieval_q, k, qfilter, need_body=True)
    if config.ENABLE_BODY_RESCUE:
        reranked = await body_rescue(retrieval_q, reranked)
    # A category facet resolved from the query (dealtype/industry) already scopes
    # results to the requested topic, so the cross-encoder score only ranks within
    # an on-topic set — don't reject those matches as "weakly related".
    faceted = bool(dealtype or industry)
    gate = config.ASK_MIN_SCORE_FACETED if faceted else config.ASK_MIN_SCORE
    sources = [s for s in reranked if s.score >= gate][: k]

    if not faceted:
        note = (
            weak_results_note([s.score for s in sources], date_label(eff_from, eff_to))
            if config.ENABLE_WEAK_FALLBACK
            else None
        )
    else:
        note = None

    if not sources:
        return PreparedTurn(answer="No sufficiently relevant articles were found for this query.", sources=[], note=note)

    if not faceted and config.ENABLE_WEAK_FALLBACK and results_are_weak([s.score for s in sources]):
        return PreparedTurn(
            answer=fallback_answer(question, len(sources), date_label(eff_from, eff_to)),
            sources=[to_summary(s).model_dump() for s in sources],
            note=note,
        )

    # As the source count grows (a 'top N' request), trim each source's body
    # excerpt so the total stays within the budget: more articles to rank at the
    # same token cost, without blowing the LLM context window.
    body_limit = min(config.CHAT_BODY_CHAR_LIMIT, config.CHAT_TOTAL_BODY_CHARS // max(1, len(sources)))
    context = "\n\n".join(source_context(s, i + 1, body_limit=body_limit) for i, s in enumerate(sources))
    history_text = "\n".join(f"{m.role}: {m.content}" for m in history if m.role in ("user", "assistant"))
    prompt = CHAT_PROMPT.format(
        history=history_text or "(none)",
        context=context,
        question=question,
        dataviz_max_rows=k,
        dataviz_view_instruction=_dataviz_view_instruction(question),
    )
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
    result = await _answer_ranked(question, turn.answer)
    await record_cost(result.cost())
    return (
        _finalize_answer(result.content, question),
        turn.sources,
        turn.note,
        result.prompt_tokens,
        result.completion_tokens,
        result.cost(),
    )


def state_llm():
    from app import main

    return main.state.get("llm")


CHAT_PROMPT = """You are Ask VCCircle, an assistant that answers questions using VCCircle's article \
database. Cite the article number(s) for every factual claim, like [1] or [2][3]. If the user asks a \
follow-up question, use the conversation history for context, but only make claims supported by the \
articles. If the articles contain no relevant information, say so plainly instead of guessing.

## Ranked "top N" lists
When asked for a ranked list (e.g. "top 10 IPO deals in 2025", "top 15 deals", "biggest funding rounds"):
- Build the list only from items the articles actually name (deals, companies, rounds, amounts).
- List as many distinct items as the articles support, up to N. If fewer than N are supported, list \
those and say you found fewer than N.
- Order by significance (highest value / biggest impact first), citing the article for each item.
- Never invent an item no article names.
- **Never refuse, and never say a ranked list "cannot be generated"**, just because the articles lack \
exact values or a pre-made ranking. If the articles name the items but don't state amounts, rank them \
by whatever is known — prominence, size, or recency — and write "value not stated" for each unknown \
value. A ranked list of the named items (even with every value "not stated") always beats a refusal.
- This applies to EVERY ranked/numeric list question, not just IPOs: funding rounds, deals, M&A, \
stake sales, companies, funds raised, hires — all of them.

## IPO-specific questions
For questions about IPOs or public listings ("top IPOs of 2025", "table of top 10 IPOs"), the list items \
are COMPANIES that went public or filed for an IPO — never private funding rounds, stake sales, or M&A. \
Do not substitute other deal types. Include IPOs with no disclosed proceeds; write "value not stated" \
rather than dropping them, and never refuse the ranked list for want of proceeds data — rank the named \
IPO companies by prominence/recency and mark each missing value "not stated".

## Charts and tables (only when explicitly requested)
Only when the user explicitly asks for a chart, graph, plot, diagram, or table/visual view (e.g. "show \
me a chart", "bar chart", "as a table"), end your answer with exactly ONE JSON block in a fenced code \
block tagged `dataviz`:

```dataviz
{{"title": "Top 2025 deals", "columns": ["Deal", "Value ($B)"], "rows": [["Zepto raise", 1.0], ["Shriram Finance stake", 4.4]], "value_column": 1, "format": "$B"}}
```

Variants:
- **Share breakdown**: percentage column, `"format": "%"` (e.g. columns `["Segment", "Share (%)"]`).
- **Year-over-year trend**: year in the first column (e.g. `["Year", "Deals"]`, rows like `["2021", 120]`).
- Optionally include `"kind"`: `"bar"`, `"line"`, or `"pie"` — use `"line"` for trends, `"pie"` for share breakdowns.

{dataviz_view_instruction}

**Data block rules:**
- Rows are the ranked items (max {dataviz_max_rows}); every item mentioned in your prose answer must \
appear as a row.
- Do NOT render the same data as a markdown table in the prose — the data block IS the table. Keep \
the prose as a short summary with citations.
- The first (label) column must contain each item's name (deal, company, round, year, segment) — \
never leave a label cell empty; a row of bare numbers is useless to the user.
- Every value cell is a plain number in the unit declared by `"format"` (`"$B"`, `"$M"`, `"₹ Cr"`, `"%"`, or `""`).
- If a value isn't stated in the articles, use `""` for that cell (never the text "value not stated" — \
that phrasing is for prose only) and never drop the row.
- If NO item has a stated value, still emit the block with item names plus a status column whose cells \
are `"not stated"`, and set `"value_column"` to `null`.
- Only include numbers actually stated in the articles — never invented ones.
- Keep `[n]` citations only in the prose, never inside the data block.
- Omit the block entirely unless the user explicitly asked for a chart, graph, plot, diagram, or table/visual view.

## Grounding discipline
- If two articles conflict on a fact (e.g. different deal values), surface both with their citations \
rather than silently picking one.
- Keep answers concise by default; expand only as far as the articles support.

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
async def create_session(request: Request):
    user_id = request.state.user_id
    return await _require_store().create_session(user_id)


@router.get("/sessions", response_model=list[SessionOut])
async def list_sessions(request: Request):
    user_id = request.state.user_id
    return await _require_store().list_sessions(user_id)


@router.get("/sessions/{session_id}", response_model=SessionDetailOut)
async def get_session(session_id: str, request: Request):
    user_id = request.state.user_id
    s = _require_store()
    session = await s.get_session(session_id, user_id)
    if session is None:
        raise HTTPException(status_code=404, detail="conversation not found")
    messages = await s.messages(session_id, user_id)
    return SessionDetailOut(**session.model_dump(), messages=messages)


@router.patch("/sessions/{session_id}", response_model=SessionOut)
async def rename_session(session_id: str, body: MessageIn, request: Request):
    user_id = request.state.user_id
    return await _require_store().rename_session(session_id, user_id, body.content)


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str, request: Request):
    user_id = request.state.user_id
    await _require_store().delete_session(session_id, user_id)
    return {"ok": True}


@router.get("/usage", response_model=SessionStatsOut)
async def get_usage(request: Request):
    """Aggregate token usage and cost across the user's conversations."""
    user_id = request.state.user_id
    return await _require_store().stats(user_id)


@router.post("/sessions/{session_id}/messages", response_model=TurnOut)
async def send_message(session_id: str, body: MessageIn, request: Request):
    user_id = request.state.user_id
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
async def send_message_stream(session_id: str, body: MessageIn, request: Request):
    """SSE-streamed chat turn: retrieves, then streams the LLM answer token by
    token. Events: 'start', 'delta' (content chunk), 'done' (with full message,
    sources, usage, cost, latency), or 'error'. The assistant message is saved
    once streaming completes."""
    user_id = request.state.user_id
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
            prompt_tokens = usage.prompt_tokens if usage else 0
            completion_tokens = usage.completion_tokens if usage else 0
            if parse_dataviz(answer) is None and _CHART_INTENT_RE.search(question):
                # The user explicitly asked for a chart/graph/plot/table but the
                # answer streamed without one; ask once more so visual requests
                # reliably carry a dataviz block. A failed retry keeps the
                # streamed answer instead of erroring the whole turn after the
                # user already saw it stream in.
                try:
                    nudge = await generate_answer(state_llm(), turn.answer + _dataviz_nudge(question), config.LLM_MODEL)
                except LLMUnavailableError:
                    nudge = None
                if nudge is not None:
                    answer = nudge.content
                    prompt_tokens += nudge.prompt_tokens
                    completion_tokens += nudge.completion_tokens
            if _is_ranking_question(question) and _is_ranking_refusal(answer):
                # A ranked/numeric list question streamed back as a refusal
                # ("cannot be generated", "no specific amounts"). Ask once more
                # to rank the named items with "value not stated" for unknowns.
                try:
                    nudge = await generate_answer(state_llm(), turn.answer + _RANKING_NUDGE, config.LLM_MODEL)
                except LLMUnavailableError:
                    nudge = None
                if nudge is not None:
                    answer = nudge.content
                    prompt_tokens += nudge.prompt_tokens
                    completion_tokens += nudge.completion_tokens
            answer = _finalize_answer(answer, question)
            result = LLMResult(
                content=answer,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
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