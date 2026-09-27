# VCCircle New Search — API Reference

Base URL: `http://<host>/` — the public entrypoint is nginx on port 80 (plain
HTTP; the app is not served on an internal port). The FastAPI backend binds
`127.0.0.1:8001` on the host itself, so it answers on `http://127.0.0.1:8001`
for internal/dev use only and is not reachable from off-host at all.

Most endpoints return JSON. Search and analytics are `GET`; chat is JSON or
Server-Sent-Events (SSE).

**Authentication:** chat, analytics and user-management endpoints require a
bearer token issued by `POST /api/auth/login`
(`Authorization: Bearer <token>`). Tokens are opaque, expire after
`AUTH_TOKEN_TTL_DAYS` (7) and can be revoked (`POST /api/auth/logout`). Access
is role-based: `user` (the only role public signup can grant — it is not
configurable) may use chat; `admin` also has
analytics read + user management. `/search`, `/facets`, `/analytics/click` and
the auth endpoints are public. Signup/login are rate-limited per IP and, for
login, per submitted address counting failed attempts only (a correct password
is never rate-limited); all inputs are validated server-side.
Internal machine clients may authenticate with `X-Service-Token` (config
`AUTH_SERVICE_TOKEN`) — a scoped, expiring credential, not an open admin grant.

### Auth endpoints

| Method & path | Access | Purpose |
|---|---|---|
| `POST /api/auth/signup` | public | Create account → `{message}` (no token; then log in) |
| `POST /api/auth/login` | public | Exchange email+password → `{token, user}` |
| `GET /api/auth/me` | auth | Current user profile |
| `POST /api/auth/logout` | auth | Revoke current token |
| `POST /api/auth/change-password` | auth | Change password; revokes other tokens → `{token, user}` |
| `GET /api/auth/users` | `admin` | List users |
| `GET /api/auth/users/{id}` | `admin` | User detail |
| `PATCH /api/auth/users/{id}` | `admin` | Update name/role/is_active |
| `DELETE /api/auth/users/{id}` | `admin` | Delete user + revoke tokens |
| `POST /api/auth/users/{id}/tokens/revoke` | `admin` | Revoke all of a user's tokens |

Signup validation: `email` (format, ≤254, lowercased),
`password` (8–128 chars, must contain a letter and a digit), `name` (optional,
≤60, no control characters). Signup always returns `200`
`{"message": "If this email is not already registered, your account is ready. Sign in with your email and password to continue; if you already have an account, sign in with your existing password."}`
and never a token, so a fresh address and an already-registered one (including
the concurrent-duplicate race) are indistinguishable — no account enumeration.
Get a token by logging in. Login returns an identical generic `401` for
unknown email or wrong password (no account enumeration). A disabled account
(`is_active=false`) is rejected everywhere.

Admin endpoints are gated by granular permissions: `analytics:read`
(`/analytics/*`), `users:read` (`GET /api/auth/users...`), and `users:manage`
(`PATCH/DELETE /api/auth/users...` and token revocation). The default `admin`
role carries all three.

---

## Integration guide (for the external consumer)

### 1. Auth flow

1. **Sign up** (`POST /api/auth/signup`), then **log in**
   (`POST /api/auth/login`) — signup returns only
   `{ "message": "..." }`, and login returns
   `{ "token": "<opaque bearer token>", "user": {...} }`.
2. Send the token on every protected request:
   `Authorization: Bearer <token>`.
3. Tokens expire after `AUTH_TOKEN_TTL_DAYS` (7). When a call returns `401`,
   re-authenticate with `POST /api/auth/login` to mint a fresh token — do not
   try to refresh an expired token. `POST /api/auth/logout` revokes the current
   token server-side (treat the client copy as dead after that).
4. Store tokens server-side only if your client needs to act on behalf of users;
   otherwise log in per user session. Never store raw passwords or tokens in
   browser-side JavaScript that ships to visitors.

### 2. Protection matrix

| Access | Endpoints |
|---|---|
| **Public** (no token) | `GET /search`, `GET /facets`, `POST /analytics/click`, `POST /api/auth/signup`, `POST /api/auth/login`, health (`/health`, `/live`, `/ready`, `/readyz`) |
| **Any authenticated user** (`user` role, default) | `POST/GET/PATCH/DELETE /api/chat/...`, `POST /api/auth/logout`, `POST /api/auth/change-password` |
| **Admin only** | `GET /analytics/summary`, `GET /analytics/chat`, all `GET/PATCH/DELETE /api/auth/users...` |

Chat conversations are scoped to the account that created them — a token can
never see or modify another account's conversations. `403` means the token is
valid but the role is not allowed; `401` means missing/expired/revoked token.

### 3. Rate limits (per IP, Redis)

- Signup: `AUTH_SIGNUP_RATE_PER_MIN` (5) — exceed → `429`.
- Login: `AUTH_LOGIN_RATE_PER_MIN` (10) — exceed → `429`.
- `/search` and chat are not IP-rate-limited (chat is bounded by the global LLM
  daily budget instead).

### 4. Consuming the chat SSE stream

`POST /api/chat/sessions/{id}/messages/stream` returns Server-Sent Events.
Read the body as a stream, split on blank lines, and parse `data:` lines as
JSON per `event:` line:

| event | payload |
|---|---|
| `start` | `{ "user": Message }` |
| `delta` | `{ "text": "..." }` — append to the answer |
| `done` | `{ "message": Message, "note": string\|null, "latency_ms": number }` — final, persisted message (sources, usage, cost) |
| `error` | `{ "error": "..." }` — nothing persisted |

If the stream drops mid-turn, the turn was not saved; re-send to retry.
Ranked-list / numeric / breakdown answers may end with a fenced
```` ```dataviz ```` JSON block (see the Dataviz section) — render it as a
chart or strip the fence before showing raw markdown.

### 5. Errors & conventions

- Errors are uniform JSON: `{"detail": "<message>"}` (FastAPI default).
- Status codes: `401` auth required/expired, `403` role forbidden, `404` session
  not found, `409` duplicate email, `422` input validation, `429` rate limit or
  daily LLM budget reached, `503` LLM/model unavailable.
- All `GET /search` responses carry `cached`, `latency_ms`, and `note` fields.
- The internal eval scripts authenticate with an `X-Service-Token` header. It is
  a scoped, expiring credential: it may only exercise
  `AUTH_SERVICE_TOKEN_SCOPE` (default `chat:use`) and stops working
  `AUTH_SERVICE_TOKEN_MAX_AGE_SECONDS` (default 24h) after it is first seeded. A
  restart does **not** revive it. To rotate, an admin mints a replacement with
  `POST /api/auth/service-tokens`, moves consumers onto it, then retires the old
  one with `POST /api/auth/service-tokens/revoke` and a body of
  `{"token": "<old value>"}` (an empty body revokes all of them at once — the
  right move for a suspected leak, the wrong one for a planned rotation, since
  it would take the replacement down too). A long-running machine client that
  runs past 24h must be given a rotated value. Never expose it in a browser
  client.

---

## `GET /search`

Hybrid semantic search (dense + sparse BM25, RRF-fused, reranked). No LLM involved.

### Query parameters

| Param       | Type   | Required | Default | Notes |
|-------------|--------|----------|---------|-------|
| `q`         | string | yes      | —       | Free-text query (`1..512` chars, `SEARCH_QUERY_MAX_CHARS`; longer → `422`) |
| `top_k`     | int    | no       | `8`     | Result count, `1..50` |
| `industry`  | string | no       | —       | Comma-separated industry values (filter) |
| `dealtype`  | string | no       | —       | Comma-separated deal-type values (filter) |
| `author`    | string | no       | —       | Comma-separated author names (filter) |
| `from_date` | string | no       | —       | `YYYY-MM-DD`, inclusive |
| `to_date`   | string | no       | —       | `YYYY-MM-DD`, inclusive (end of day) |

**Query intent handling** (automatic):
- Relative/absolute years are parsed into a date filter ("last year" = previous calendar year, "2025" = that year).
- `top N <topic> in <year>` queries are rewritten to surface year-review ("Flashback <year>") articles **and** the bare topic is searched too; candidates are merged and reranked once.
- `top N <topic>` queries also raise the effective result count up to `N` (bounded by 50), so the response can actually return `N` results even when the `top_k` param is smaller.
- Query expansion maps user vocabulary to corpus terms (e.g. "layoffs" → "job cuts", "fundraise", etc.).
- Entity-mention boosting raises results that name the query's company.
- Results are additionally recency-tempered, de-duplicated by greedy MMR, and (once a query accumulates enough clicks) boosted toward articles users actually open.

### Response

```json
{
  "query": "top 10 fintech deals in 2025",
  "results": [
    {
      "id": 12345,
      "title": "Flashback 2025: Top technology M&As and PE/VC deals of the year",
      "url": "https://www.vccircle.com/...",
      "published_date": "2025-12-31T00:00:00+00:00",
      "category": "Others",
      "summary": "A year-end roundup of the biggest deals...",
      "author_names": ["Priya Sharma"],
      "industry_names": ["Fintech"],
      "dealtype_names": ["M&A"],
      "score": 0.998
    }
  ],
  "cached": false,
  "latency_ms": 214.3,
  "note": null
}
```

`note` is a human-readable hint when results are only weakly related to the query
(otherwise `null`). `score` is the cross-encoder reranked relevance in `0..1`
(entity-boosted results can exceed `1.0`). `published_date` may be `null`.

---

## Chat API (per-user conversations)

Conversations are stored per authenticated account in SQLite and survive
restarts; they are purged after `CHAT_RETENTION_DAYS` (180) of inactivity.
**Every chat request must send `Authorization: Bearer <token>`** (or the
`X-Service-Token` machine credential). Conversations are scoped to the account, so
other users can never see or modify them.

### Identity & session shape

`Session` (list/create/get/rename):

```json
{
  "id": "f0e1d2c3...",
  "title": "Who invested in Ola Electric?",
  "created_at": 1786950000.0,
  "updated_at": 1786953600.0,
  "last_preview": "first 140 chars of the last message",
  "total_cost": 0.3017
}
```

`Message`:

```json
{
  "id": 42,
  "role": "assistant",
  "content": "Ola Electric raised... [1]",
  "sources": [ { "id": 53671, "title": "...", "url": "https://www.vccircle.com/...", "published_date": "2023-01-04T12:17:39+00:00", "category": "Others", "score": 0.91 } ],
  "created_at": 1786953600.0,
  "prompt_tokens": 2065,
  "completion_tokens": 323,
  "cost": 0.1369,
  "latency_ms": 1800.0
}
```

### Endpoints

| Method & path | Purpose |
|---|---|
| `POST /api/chat/sessions` | Create a conversation → `Session` |
| `GET /api/chat/sessions` | List the user's conversations (newest first, up to 100) → `Session[]` |
| `GET /api/chat/sessions/{id}` | Fetch a conversation + full message list → `Session` with `messages: Message[]` (404 if not owned) |
| `PATCH /api/chat/sessions/{id}` | Rename; body `{ "content": "new title" }` → `Session` |
| `DELETE /api/chat/sessions/{id}` | Delete the conversation (cascades messages) → `{"ok": true}` |
| `GET /api/chat/usage` | Per-user aggregates → `{"sessions", "messages", "total_tokens", "total_cost"}` |
| `POST /api/chat/sessions/{id}/messages` | One-shot turn (non-streaming) → `TurnOut` (below) |
| `POST /api/chat/sessions/{id}/messages/stream` | SSE-streamed turn → see below |

### `POST /api/chat/sessions/{id}/messages`

Body: `{ "content": "who invested in Ola Electric?" }` (max 8000 chars).

Response (`TurnOut`):

```json
{
  "user": { "id": 41, "role": "user", "content": "who invested in Ola Electric?", "sources": [], "created_at": ..., "prompt_tokens": 0, "completion_tokens": 0, "cost": 0, "latency_ms": 0 },
  "assistant": { "id": 42, "role": "assistant", "content": "...", "sources": [...], "prompt_tokens": ..., "completion_tokens": ..., "cost": ..., "latency_ms": ... },
  "note": null,
  "latency_ms": 1800.0
}
```

Both the user message and the assistant reply (with sources, tokens, cost and
latency) are persisted. Greetings/small talk are answered without an LLM call;
turns with only weak results get an honest fallback reply (zero tokens/cost).
When the top-ranked sources score weakly, the backend re-scores them against
the most relevant region of their article bodies ("body rescue"), so facts that
live mid-article (e.g. historical retrospectives) can still pass the relevance
gate and be cited.

### `POST /api/chat/sessions/{id}/messages/stream` (SSE)

Same body/headers as the one-shot endpoint. Returns `text/event-stream` with
named events:

| Event | Payload | Meaning |
|---|---|---|
| `start` | `{ "user": Message }` | User message persisted |
| `delta` | `{ "text": "..." }` | One streamed content chunk of the answer |
| `done` | `{ "message": Message, "note": string\|null, "latency_ms": number }` | Answer finished; `message` is persisted (sources/usage/cost filled in) |
| `error` | `{ "error": "..." }` | Turn failed (e.g. LLM unreachable); nothing persisted |

Example consumption:

```bash
TOKEN=$(curl -s -X POST http://localhost:8001/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"you@example.com","password":"secret12"}' | jq -r .token)
curl -N -X POST http://localhost:8001/api/chat/sessions/abc/messages/stream \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"content":"who invested in Ola Electric?"}'
```

```text
event: start
data: {"user":{...}}

event: delta
data: {"text":"Ola Electric raised"}

event: delta
data: {"text":" a Series E round"}

event: done
data: {"message":{...,"prompt_tokens":2065,"completion_tokens":323,"cost":0.1369,...},"note":null,"latency_ms":1800.0}
```

### Dataviz data block (in `Message.content`)

When the user explicitly asks for a chart, graph, plot, diagram, or a
table/visual view, the assistant's `content` (in both the one-shot `TurnOut`
and the SSE `done` message) ends with **one** fenced JSON block tagged `dataviz`
that the UI renders as a Table/Bar/Line/Pie/Pictogram chart. Ranked-list /
numeric-comparison / breakdown questions that do NOT mention a visual view
answer in plain prose with no block:

```markdown
Prose answer with inline citations [1][2].

```dataviz
{"title": "Top 2025 deals", "columns": ["Deal", "Value ($B)"], "rows": [["Zepto raise", 1.0], ["Shriram Finance stake", 4.4]], "value_column": 1, "format": "$B", "kind": "bar"}
```
```

| Field | Type | Meaning |
|---|---|---|
| `title` | string (optional) | Chart heading |
| `columns` | `string[]` | Column headers; first column is the item label |
| `rows` | `(string\|number)[][]` | One array per row, aligned with `columns` (up to the requested count, capped at `CHAT_MAX_SOURCES`) |
| `value_column` | int | Index of the numeric column the chart plots |
| `format` | string (optional) | Unit for display: `"$B"`, `"$M"`, `"₹ Cr"`, `"%"`, or `""` |
| `kind` | string (optional) | Chart hint: `"bar"`, `"line"`, or `"pie"` (frontend default view) |
| `view` | string (optional) | Pinned view the user explicitly asked for: `"table"`, `"bar"`, `"line"`, `"pie"`, or `"picto"`. When present the UI renders ONLY that view (no toggles); `"bar"`/`"line"`/`"pie"` also set `kind` |

Notes:
- Emission is non-deterministic; the backend re-calls the LLM once with a
  nudge when an explicit chart request returns without a block. When the user
  asks for a specific view (e.g. "show me a pie chart", "as a table"), the
  backend pins the block's `view` to it so clients render exactly that view.
- Ranked/numeric questions that come back as a refusal ("cannot be generated",
  "no specific amounts") are re-asked once with a nudge to rank the named items
  and state "value not stated" for unknowns.
- Malformed blocks (invalid JSON, ragged rows, non-numeric value column) are
  stripped by `_sanitize_dataviz` before storage, so `content` never exposes
  unparseable JSON to clients.
- The block is judged by ONE rule shared with the frontend — `parse_dataviz` in
  `backend/app/chat.py` and `parseDataViz` in
  `frontend/app/chat/datavizContract.ts`, compared fixture by fixture by
  `backend/tests/test_dataviz_contract.py` — so a block the server keeps is
  always a block the client can render, and one it strips never reaches either.
  Two consequences: `value_column` may be written as a whole number in float
  form (`2.0` is column 2), and a value-less block is kept only for a pinned
  `table` view. A value cell counts as a number only when the whole cell is a
  plain numeric literal (`"1,200"` yes, `"12abc"` no), and never when it is
  infinite or NaN.
- Consumers should treat the block as optional and never fail to render the
  prose when it is absent.

---

## `GET /facets`

Distinct values for filter autocomplete. Cached in Redis.

### Response

```json
{
  "industry": ["Cleantech", "Consumer", "Fintech", ...],
  "dealtype": ["Credit", "M&A", "Private Equity", "Venture Capital", ...]
}
```

---

## `POST /analytics/click`

Anonymous result-click beacon sent by the frontend when a user opens a result
(no user identifiers, no cookies). `sendBeacon` from the search page.

### Body

```json
{ "query": "fintech funding", "position": 2, "id": 12345 }
```

| Field | Type | Meaning |
|---|---|---|
| `query` | string | The query string the user searched |
| `position` | int | 1-based position of the clicked result |
| `id` | int (optional) | The clicked article's id, used by click-driven learning |

### Response

`200` with `{"ok": true}`. Recording is best-effort; a Redis outage never
affects search.

---

## `GET /analytics/summary`

Aggregated search-quality and click metrics, stored in Redis DB 1
(`ANALYTICS_REDIS_DB`). Admin-only: requires a bearer token for an account with
the `analytics:read` permission (`Authorization: Bearer <token>`).

### Response

```json
{
  "searches_total": 120,
  "searches_today": 14,
  "zero_result_rate": 4.2,
  "weak_result_rate": 8.3,
  "filtered_rate": 12.5,
  "cache_hit_rate": 61.0,
  "avg_latency_ms": 214.3,
  "clicks_total": 33,
  "top_queries": [["fintech funding", 22], ...],
  "click_positions": { "1": 12, "2": 8, ... },
  "click_top_queries": [["fintech funding", 9], ...]
}
```

Counters reset when the analytics Redis DB is cleared (`redis-cli -n 1 FLUSHDB`).

`click_positions` is keyed by result position from `CLICK_POSITION_MIN` to
`CLICK_POSITION_MAX` (`1`..`10` in this codebase; both are constants in
`backend/app/analytics.py`, not environment settings). A click recorded outside
that range is clamped to the nearest bound, so every reported bucket is one the
backend can record. `top_queries` returns at most `TOP_QUERIES_N` (20) entries
and `click_top_queries` at most `TOP_CLICKED_QUERIES_N` (10).

---

## `GET /analytics/chat`

Cross-user chat usage, read from the SQLite chat store. Admin-only
(`analytics:read`), same gate as `/analytics/summary`.

### Response

```json
{
  "sessions": 21,
  "users": 12,
  "messages": 55,
  "total_tokens": 61397,
  "total_cost": 2.1901243,
  "avg_latency_ms": 1797.1,
  "top_by_cost": [ ["<session-id>", 4, 0.3017, 1786954406.95], ... ],
  "top_by_tokens": [ ["<session-id>", 6, 8432, 1786956978.24], ... ],
  "sessions_today": 3,
  "daily_sessions": [ ["2026-08-17", 3], ... ]
}
```

This is a cross-user response and it is **not** content-free: it exposes
global totals plus per-session rows for every user's conversations. What keeps
it free of user-authored text is that no session title, message body or any
other user-written string is ever selected — the top-N queries project
`sessions.id` only, so a session is identified by its opaque id and nothing
else. (A session title is the first 60 characters of the user's own question,
so returning one here would hand every admin the opening of other people's
private conversations.)

Each `top_by_*` row is `[session_id, messages, cost | tokens, updated_at]`.

Every read is recorded in the durable `admin_audit` table (`actor_id`,
`action`, `created_at`), written by `ChatStore.record_admin_audit` before the
response is returned. Rows older than `AUDIT_RETENTION_DAYS` (90) are dropped by
the existing retention sweep, `ChatStore.purge_expired`, so recording a read
stays a single INSERT despite this endpoint being polled every 30s.

**Reading the trail.** `ChatStore.admin_audit_log` is the reader, and it is a
store method with no HTTP surface — recovering the trail means querying the
chat SQLite database directly (or calling the method from a Python shell),
which needs filesystem access. There is no admin UI or endpoint for it. That
is a deliberate position, not an oversight: publishing cross-user read history
over the API would create a second cross-user disclosure, in the endpoint
this issue exists to harden. Be aware of the cost — a control nobody can
easily read deters less than it appears to.

**What the trail does and does not establish.** It answers "which admin read
cross-user chat analytics, when, and how often". It does *not* support
detecting a slow browse through individual conversations: `action` is a
constant and no row records which sessions were returned, so a deliberate
browse and an idle open dashboard tab look identical. Per-subject attribution
was deliberately omitted rather than overlooked — it would put other users'
session ids into the audit table, trading this fix's own privacy goal for a
weaker signal.

`actor_id` is the authenticated account's id, so human admin logins are
attributed individually. A request authenticated with the shared
`X-Service-Token` bypass is recorded under the single `SERVICE_USER_ID`
constant instead — the trail cannot distinguish callers that present the same
shared secret, and no code change can recover a per-caller identity from one
secret. Treat machine-bypass reads as attributable to "the service token",
not to a person.

---

## `GET /analytics/dashboard` (frontend page)

The dashboard UI is a Next.js page at `/analytics/dashboard` (proxied by nginx
to the frontend; not part of this API). It renders KPI cards for search quality
+ chat usage, top-query tables, clicks-by-position and
conversations-by-cost/tokens tables by calling the two admin-gated JSON
endpoints below with the bearer token, and refreshes every 30s.

---

## Health endpoints

| Endpoint | Purpose | Status |
|----------|---------|--------|
| `GET /health` | Liveness | always `200` if the process is up |
| `GET /live`   | Liveness alias | `200` |
| `GET /ready`  | Readiness (JSON report) | `200` when ready, `503` if Qdrant/models unavailable |
| `GET /readyz` | Readiness (bare) | `200` / `503` |

`/ready` example:

```json
{
  "ready": true,
  "checks": {
    "qdrant": { "ok": true },
    "models": { "ok": true },
    "redis": { "ok": true, "cache": "redis" },
    "llm": { "ok": true }
  }
}
```

Redis down does not fail readiness (the API degrades to an in-process cache).

---

## Examples

```bash
# Basic search
curl "http://<host>/search?q=fintech%20funding&top_k=5"

# Filter by industry + date range
curl "http://<host>/search?q=funding&industry=Fintech,Healthtech&from_date=2024-01-01&to_date=2025-12-31"

# Year-in-review / top-N (auto Flashback handling)
curl "http://<host>/search?q=top%2010%20fintech%20deals%20in%202025&top_k=10"

# Facet values for filter autocomplete
curl "http://<host>/facets"

# Sign up (public; rate-limited per IP). Responds 200 {"message": "..."} for a
# new AND an already-registered address, and issues no token — log in below.
curl -X POST "http://<host>/api/auth/signup" -H "Content-Type: application/json" \
  -d '{"email":"you@example.com","password":"secret12","name":"You"}'

# Log in and capture a bearer token
TOKEN=$(curl -s -X POST "http://<host>/api/auth/login" -H "Content-Type: application/json" \
  -d '{"email":"you@example.com","password":"secret12"}' | jq -r .token)

# Create a chat conversation
curl -X POST "http://<host>/api/chat/sessions" -H "Authorization: Bearer $TOKEN"

# List conversations
curl "http://<host>/api/chat/sessions" -H "Authorization: Bearer $TOKEN"

# Per-user token/cost usage
curl "http://<host>/api/chat/usage" -H "Authorization: Bearer $TOKEN"

# Stream a chat turn (SSE)
curl -N -X POST "http://<host>/api/chat/sessions/<id>/messages/stream" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"content":"who invested in Ola Electric?"}'
```

---

## Notes & limitations

- **Auth**: bearer tokens (7-day expiry, revocable) gate chat, analytics and
  user management; `/search`, `/facets`, `/analytics/click` and the auth
  endpoints are public. Signup/login are rate-limited per IP via Redis and login
  additionally per submitted address, counting failed logins only so the
  per-address limit cannot be used to lock a known account out. The
  per-address limit caps the *rate* of attempts on one account, not an
  attacker's cost — it is checked after the password verify, so being refused
  is free; the per-IP limit is what bounds cost. When Redis is unreachable
  these limits fall back to a bounded in-process limiter rather than switching
  off. A user
  holds at most `AUTH_MAX_ACTIVE_TOKENS_PER_USER` active tokens; logging in
  past that revokes the oldest. `AUTH_SERVICE_TOKEN` lets internal scripts
  authenticate as a scoped, expiring machine user.
- **Data freshness**: the index is refreshed by an incremental sync every 15 minutes
  via cron (`update_index.py`).
- **Caching**: `/search` responses are cached (TTL `CACHE_TTL_SECONDS`, default 300s) keyed by effective query + filters. `cached: true` indicates a cache hit. Chat turns are not cached. When Redis is unreachable, the cache degrades to an in-process store so the API keeps working.
- **Retention**: conversations idle for 180 days are purged daily.
- **No interactive docs**: `/docs`, `/redoc` and `/openapi.json` are disabled in
  every environment. They publish the full route list, the request/response
  models (including the mass-assignable `UserPatchIn`) and which routes sit
  behind which dependency, to any unauthenticated caller. This document is the
  API reference. To regenerate the machine-readable schema locally, run
  `python -c "import json; from app.main import app; print(json.dumps(app.openapi(), indent=2))"`
  from `backend/` and treat the output as a local artefact — do not serve it.
- **Host header**: requests whose `Host` is not in `ALLOWED_HOSTS` are rejected
  with 400 by `TrustedHostMiddleware`. The default list is derived from
  `CORS_ORIGINS`, this box's own hostname and addresses (including its
  default-route address, i.e. the one a public client is reaching) and
  `localhost`/`127.0.0.1`/`testserver`. Set `ALLOWED_HOSTS` to the hostnames
  your deployment answers to if it is reachable under a name none of those
  cover (a separately registered public domain, for instance).
