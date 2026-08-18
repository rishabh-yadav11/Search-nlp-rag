# VCCircle New Search — API Reference

Base URL: `https://<host>/` (also reachable directly at `http://<host>:8001` on the
host itself; the public entrypoint is nginx on port 80).

Most endpoints return JSON. Search and analytics are `GET`; chat is JSON or
Server-Sent-Events (SSE).

**Authentication:** chat, analytics and user-management endpoints require a
bearer token issued by `POST /api/auth/signup` or `POST /api/auth/login`
(`Authorization: Bearer <token>`). Tokens are opaque, expire after
`AUTH_TOKEN_TTL_DAYS` (7) and can be revoked (`POST /api/auth/logout`). Access
is role-based: `user` (public-signup default) may use chat; `admin` also has
analytics read + user management. `/search`, `/facets`, `/analytics/click` and
the auth endpoints are public. Signup/login are rate-limited per IP (Redis);
all inputs are validated server-side. Internal machine clients may bypass via
`X-Service-Token` (config `AUTH_SERVICE_TOKEN`).

### Auth endpoints

| Method & path | Access | Purpose |
|---|---|---|
| `POST /api/auth/signup` | public | Create account → `{token, user}` |
| `POST /api/auth/login` | public | Exchange email+password → `{token, user}` |
| `GET /api/auth/me` | auth | Current user profile |
| `POST /api/auth/logout` | auth | Revoke current token |
| `POST /api/auth/change-password` | auth | Change password; revokes other tokens → `{token, user}` |
| `GET /api/auth/users` | `admin` | List users |
| `GET /api/auth/users/{id}` | `admin` | User detail |
| `PATCH /api/auth/users/{id}` | `admin` | Update name/role/is_active |
| `DELETE /api/auth/users/{id}` | `admin` | Delete user + revoke tokens |
| `POST /api/auth/users/{id}/tokens/revoke` | `admin` | Revoke all of a user's tokens |

Signup validation: `email` (format, ≤254, lowercased, unique → 409),
`password` (8–128 chars, must contain a letter and a digit), `name` (optional,
≤60, no control characters). Login returns an identical generic `401` for
unknown email or wrong password (no account enumeration). A disabled account
(`is_active=false`) is rejected everywhere.

---

## Integration guide (for the external consumer)

### 1. Auth flow

1. **Sign up** (`POST /api/auth/signup`) or **log in** (`POST /api/auth/login`).
   Both return `{ "token": "<opaque bearer token>", "user": {...} }`.
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
- The internal eval scripts authenticate with an `X-Service-Token` header that
  acts as an admin. That bypass is for machine-to-machine use only — never
  expose it in a browser client.

---

## `GET /search`

Hybrid semantic search (dense + sparse BM25, RRF-fused, reranked). No LLM involved.

### Query parameters

| Param       | Type   | Required | Default | Notes |
|-------------|--------|----------|---------|-------|
| `q`         | string | yes      | —       | Free-text query (min 1 char) |
| `top_k`     | int    | no       | `8`     | Result count, `1..50` |
| `industry`  | string | no       | —       | Comma-separated industry values (filter) |
| `dealtype`  | string | no       | —       | Comma-separated deal-type values (filter) |
| `author`    | string | no       | —       | Comma-separated author names (filter) |
| `from_date` | string | no       | —       | `YYYY-MM-DD`, inclusive |
| `to_date`   | string | no       | —       | `YYYY-MM-DD`, inclusive (end of day) |

**Query intent handling** (automatic):
- Relative/absolute years are parsed into a date filter ("last year" = previous calendar year, "2025" = that year).
- `top N <topic> in <year>` queries are rewritten to surface year-review ("Flashback <year>") articles **and** the bare topic is searched too; candidates are merged and reranked once.
- Query expansion maps user vocabulary to corpus terms (e.g. "layoffs" → "job cuts", "fundraise", etc.).
- Entity-mention boosting raises results that name the query's company.

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
`X-Service-Token` machine bypass). Conversations are scoped to the account, so
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

For ranked-list / numeric-comparison / breakdown questions, the assistant's
`content` (in both the one-shot `TurnOut` and the SSE `done` message) ends with
**one** fenced JSON block tagged `dataviz` that the UI renders as a
Table/Bar/Line/Pie/Pictogram chart:

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
| `rows` | `(string\|number)[][]` | One array per row, aligned with `columns` (max 10 rows) |
| `value_column` | int | Index of the numeric column the chart plots |
| `format` | string (optional) | Unit for display: `"$B"`, `"$M"`, `"₹ Cr"`, `"%"`, or `""` |
| `kind` | string (optional) | Chart hint: `"bar"`, `"line"`, or `"pie"` (frontend default view) |

Notes:
- Emission is non-deterministic; the backend re-calls the LLM once with a
  nudge when a dataviz-intent question returns without a block.
- Malformed blocks (invalid JSON, ragged rows, non-numeric value column) are
  stripped by `_sanitize_dataviz` before storage, so `content` never exposes
  unparseable JSON to clients.
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
{ "query": "fintech funding", "position": 2 }
```

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
  "top_by_cost": [ ["Who invested in Ola Electric?", 4, 0.3017, 1786954406.95], ... ],
  "top_by_tokens": [ ["top deals of 2025", 6, 8432, 1786956978.24], ... ],
  "sessions_today": 3,
  "daily_sessions": [ ["2026-08-17", 3], ... ]
}
```

No message contents are exposed — only counts, totals and per-conversation
aggregates (privacy-safe).

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
curl "https://<host>/search?q=fintech%20funding&top_k=5"

# Filter by industry + date range
curl "https://<host>/search?q=funding&industry=Fintech,Healthtech&from_date=2024-01-01&to_date=2025-12-31"

# Year-in-review / top-N (auto Flashback handling)
curl "https://<host>/search?q=top%2010%20fintech%20deals%20in%202025&top_k=10"

# Facet values for filter autocomplete
curl "https://<host>/facets"

# Sign up (public; rate-limited per IP)
curl -X POST "https://<host>/api/auth/signup" -H "Content-Type: application/json" \
  -d '{"email":"you@example.com","password":"secret12","name":"You"}'

# Log in and capture a bearer token
TOKEN=$(curl -s -X POST "https://<host>/api/auth/login" -H "Content-Type: application/json" \
  -d '{"email":"you@example.com","password":"secret12"}' | jq -r .token)

# Create a chat conversation
curl -X POST "https://<host>/api/chat/sessions" -H "Authorization: Bearer $TOKEN"

# List conversations
curl "https://<host>/api/chat/sessions" -H "Authorization: Bearer $TOKEN"

# Per-user token/cost usage
curl "https://<host>/api/chat/usage" -H "Authorization: Bearer $TOKEN"

# Stream a chat turn (SSE)
curl -N -X POST "https://<host>/api/chat/sessions/<id>/messages/stream" \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"content":"who invested in Ola Electric?"}'
```

---

## Notes & limitations

- **Auth**: bearer tokens (7-day expiry, revocable) gate chat, analytics and
  user management; `/search`, `/facets`, `/analytics/click` and the auth
  endpoints are public. Signup/login are rate-limited per IP via Redis; set
  `AUTH_SERVICE_TOKEN` to let internal scripts bypass as an admin user.
- **Data freshness**: the index is refreshed by an incremental sync every 15 minutes
  via cron (`update_index.py`).
- **Caching**: `/search` responses are cached (TTL 120s) keyed by effective
  query + filters. `cached: true` indicates a cache hit. Chat turns are not cached.
- **Retention**: conversations idle for 180 days are purged daily.
- Interactive OpenAPI docs are served by FastAPI at `/docs` on the internal API port.
