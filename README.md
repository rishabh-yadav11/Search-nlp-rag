# VCCircle New Search

Hybrid retrieval + RAG search over the VCCircle article corpus, with a
ChatGPT-style chat assistant. Articles are indexed into Qdrant with dense
(semantic) and sparse (BM25) vectors fused with Reciprocal Rank Fusion (RRF). A
FastAPI service queries the index, synthesizes cited answers with an LLM (Google
Gemini via an OpenAI-compatible endpoint), and stores per-user chat
conversations in SQLite. A Next.js UI ties it together with a search page and a
`/chat` page.

```
MySQL ──fetch_data──▶ articles.jsonl ──build_index──▶ Qdrant (dense + sparse)
   │                                                         ▲
   └─────────update_index (incremental new/edit/delete)──────┘
                                                               │
Browsers ◀── nginx ───▶ Next.js app ◀──(same origin)──▶ FastAPI ──▶ Qdrant
                         /search, /chat, /facets,          │
                         /health, /live, /ready            Gemini (for chat)
                                                           Redis (cache + analytics)
                                                           SQLite (chat conversations)
```

## Repository layout

```
backend/
  app/
    main.py            FastAPI app: /search, /chat, /health, /analytics
    health.py          /live, /ready, /readyz dependency checks
    encoders.py        fastembed/ONNX dense encoder with torch fallback
    reranker.py        ONNX cross-encoder reranker with torch fallback
    llm.py             LLM call with timeout, retries, backoff, token-cost calc
    cost_budget.py     daily LLM spend cap (fails closed once budget hit)
    chat.py            per-user chat store (SQLite) + /api/chat router
    analytics.py       Redis-backed search/click analytics aggregates
    config.py          env-driven settings
    query_intent.py    year/top-N intent parsing + Flashback rewriting
    query_expand.py    deterministic synonym query expansion
    query_fix.py       SymSpell query typo correction
    diversity.py       MMR title-diversity reordering of results
    click_boost.py     click-signal score boost on results
    rerank_boost.py    entity-mention score boost on reranked results
    answer_fallback.py weak-result notes + honest chat fallback replies
    index_text.py      shared text composition + date normalization
    redis_cache.py     Redis-backed cache with in-process fallback
  scripts/
    fetch_data.py      MySQL -> data/articles.jsonl (paginated, resumable)
    build_index.py     articles.jsonl -> Qdrant embeddings (checkpointed)
    build_query_vocab.py  corpus token vocab for query_fix (gzip JSON)
    update_index.py    incremental MySQL->Qdrant sync (new/edit/delete)
    backfill_summary.py / backfill_body.py / backfill_missing.py  payload backfills
    backup_qdrant.py   Qdrant snapshot + local artifact backups (retention)
    qdrant_backup.py   shared backup helpers
    reset_index.py     drop the index + data files (backup-gated)
    search_quality_test.py / chat_quality_test.py  golden-set search/chat evals
    deep_body_eval.py  whole-body indexing + body-grounded chat eval
    rerank_bench.py    reranker latency/quality benchmark (onnx vs torch)
    load_test.py       query load/throughput harness
  tests/               pytest suite (offline, mocked deps)
  TEST_COVERAGE_GAPS.md  coverage-gap checklist (86% overall, 285 passed)
  requirements.txt
  requirements-dev.txt lint/test tooling
  .env.example         configuration template
frontend/              Next.js app (App Router + TypeScript)
  app/page.tsx         search UI (timeouts, validation, a11y)
  app/chat/page.tsx    ChatGPT-style chat UI (SSE streaming)
  app/chat/DataViz.tsx hand-rolled SVG charts (table/bar/line/pie/pictogram)
  app/analytics/dashboard/page.tsx  admin analytics dashboard
  app/login/ app/signup/  auth pages (token + RBAC)
  app/lib/auth.ts      client-side session/token helpers
  app/globals.css
  middleware.ts        CSP nonce header
  next.config.ts       security headers (CSP, nosniff, etc.)
  eslint.config.mjs    flat config for eslint 9
setup.sh               one-command deploy (deps, services, index, nginx, cron)
deploy/                healthcheck.sh (cron health probe) + logrotate.conf
docs/API.md            chat + auth API contract
.github/workflows/ci.yml   backend + frontend + security gates
```

## How it works

- **Hybrid retrieval** — every article is embedded twice at index time:
  `BAAI/bge-base-en-v1.5` (dense, cosine, 768-dim) and `Qdrant/bm25` sparse
  vectors with IDF. At query time both are searched in a single Qdrant prefetch
  and fused with RRF. The **dense** vector is `title + authors + industry +
  dealtype + summary` (metadata first, no body, short and fast to encode on
  CPU); the **sparse (BM25)** vector is the same plus the full `body`, so
  keyword matches inside article bodies stay searchable at lexical cost.
- **Faceted filtering** — `/search` accepts optional `industry`,
  `dealtype`, `author` and `from_date`/`to_date` params, applied as a Qdrant
  filter to both prefetches. Any value can be comma-separated for multi-select.
  Cache keys include the filters so distinct queries don't collide.
- **Query intent** — relative/absolute years ("last year", "2025", spans) are
  parsed into automatic date filters, and "top/best N X in Y" is rewritten to
  "Flashback Y X" for year-review retrieval with a bumped `top_k`. Explicit
  user-supplied dates always win.
- **Reranking** — RRF candidates are re-scored with a cross-encoder
  (`cross-encoder/ms-marco-MiniLM-L-6-v2`) over `title + summary`; the final
  `top_k` reflects true relevance and the score shown to the user is that
  reranked score (0-1).
- **Result ordering** — recency-tempered relevance first: blended score desc
  (`score * (1 - RECENCY_STRENGTH * (1 - exp(-age_days / RECENCY_DECAY_DAYS)))`),
  then `published_date` desc as tie-break (missing dates last).
- **Query typo correction** — `query_fix.py` runs SymSpell dictionary correction
  over a corpus-derived token vocab (built by `scripts/build_query_vocab.py`)
  plus a curated entity list (companies, VCs, people), right before embedding.
  Short tokens are capped at one edit, corrections are spliced back preserving
  punctuation, and uncorrected tokens fall through to the hybrid layer which
  absorbs mild typos. When the vocab artifact is missing the fixer is a silent
  no-op, so startup never depends on the one-time artifact.
- **Result diversity** — `diversity.py` reorders search results with greedy MMR
  over title-word Jaccard similarity, so a deal covered by several
  near-identical headlines doesn't crowd out other stories in the top-n. `lam`
  near 1 keeps relevance dominant; `sim_thresh` is a floor below which a pair's
  similarity contributes no diversity penalty.
- **Click boost** — `click_boost.py` lifts results that have earned real click
  share from the analytics Redis store (gated on minimum clicks/share), and
  re-sorts the final list. Redis down degrades to a pass-through.
- **ONNX fast paths** — query-time dense embedding uses fastembed's ONNX (INT8)
  variant of the bge model and the cross-encoder reranker runs via optimum
  ONNX; both keep the same vector directions as torch so the existing index
  stays valid, and both fall back to torch if the model can't load at startup.
- **Chat (conversations)** — a ChatGPT-style UI at `/chat`. Users sign up/log in
  (token + RBAC: public signup grants `user` = chat; the `admin` role adds
  analytics + user management; the bootstrap admin comes from
  `AUTH_ADMIN_EMAIL`/`AUTH_ADMIN_PASSWORD`). Conversations are stored per
  account in SQLite (`backend/data/chat.db`, WAL mode) so they survive restarts,
  shared across gunicorn workers. Turns reuse the same retrieval/rerank/fallback
  pipeline as search but build a conversation-aware prompt; the answer is
  streamed token-by-token over SSE (`POST .../messages/stream`) and the assistant
  message, sources, tokens, cost and latency are persisted. Conversations idle
  for `CHAT_RETENTION_DAYS` (180) are purged daily. See `docs/API.md`.
  Answers carry a ` ```dataviz ` JSON block (`{title?, columns, rows,
  value_column, format?, kind?, view?}`) that the UI renders as Table/Bar/Line/
  Pie/Pictogram charts (hand-rolled SVG, no chart deps) — but ONLY when the user
  explicitly asks for a chart/graph/plot/table (e.g. "show me a chart"), so
  ranked/numeric questions answer in plain prose by default. When the user asks
  for a specific view (table, bar/graph, line, pie, or pictogram), the block's
  `view` is pinned to it and the UI renders ONLY that view (no view toggles).
  Emission is non-deterministic, so the backend re-prompts once with a nudge
  when an explicit chart request comes back without a block, and malformed
  fences are stripped before storage so raw JSON never reaches users.
- **Caching** — Redis-backed JSON cache shared across workers (falls back to
  an in-process cache if Redis is down) for `/search`, keyed
  by effective query + top_k + facets. The retrieval/rerank step
  (`retrieve_and_rerank`) is cached on the same key space (query + filter +
  top_k), so repeated searches AND every chat turn that re-asks the same
  question skip embedding + rerank entirely.

### Health endpoints

| Endpoint | Purpose | Fails on |
|---|---|---|
| `/health` | Liveness (always 200 if the process is up) | — |
| `/live` | Liveness alias | — |
| `/ready` | Readiness, JSON report | Qdrant down or models not loaded → `503` |
| `/readyz` | Readiness, bare status code | same as `/ready` |

`/ready` checks the Qdrant collection, model/reranker loading, and reports Redis
and LLM (Gemini) status non-fatally (Redis failures degrade to the in-process
cache, so they don't flip readiness).

## Prerequisites

- Python 3.11–3.12 (warned-but-tolerated on 3.13/3.14; set `ALLOW_UNSUPPORTED_PY=1` to silence)
- Node.js 18+ (22 recommended for the frontend)
- Docker (for Qdrant and Redis)
- MySQL source database

## One-command deployment (recommended)

`setup.sh` provisions everything in stages; run `./setup.sh all` (or select
stages):

```bash
./setup.sh backend      # python check, venv + deps, Qdrant/Redis (bound to 127.0.0.1)
./setup.sh index        # fetch_data -> build_index -> seed incremental state
./setup.sh frontend     # npm ci + production build
./setup.sh services     # pm2 start gunicorn (API) + next (frontend)
./setup.sh pm2-startup  # systemd unit so services restore on reboot
./setup.sh cron         # 15-min incremental sync
./setup.sh nginx        # reverse proxy + security headers on :80
./setup.sh all          # deps backend index frontend services pm2-startup cron nginx
```

Environment overrides: `QDRANT_PORT`, `REDIS_PORT`, `API_PORT`, `NEXT_PORT`,
`PUBLIC_PORT`, `GUNICORN_WORKERS`, `PUBLIC_BASE_URL`, `QDRANT_IMAGE`,
`REDIS_IMAGE`, `ALLOW_UNSUPPORTED_PY`.

If you'd rather run pieces manually, keep reading.

## 1. Backend setup

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # fill in MySQL creds + GEMINI_API_KEY
```

Qdrant and Redis, local (mirrors what `setup.sh backend` does):

```bash
docker run -d --name qdrant -p 127.0.0.1:6333:6333 -p 127.0.0.1:6334:6334 \
  -v $(pwd)/qdrant_data:/qdrant/storage --restart unless-stopped \
  qdrant/qdrant:v1.19.0@sha256:057ee3a8da769fe7310dd3537b4dc7583bf87a95ce8ac43c0af5a46bc580d1fc
docker run -d --name redis -p 127.0.0.1:6379:6379 --restart unless-stopped redis:7-alpine
```

## 2. Build the index (run once)

```bash
cd backend
python scripts/fetch_data.py     # MySQL -> data/articles.jsonl
python scripts/build_index.py    # embed + upsert into Qdrant
```

Both are safe to interrupt and re-run: `fetch_data.py` resumes from the max id
already written; `build_index.py` resumes from a line-number checkpoint and only
advances it after Qdrant acknowledges the upsert (no silent data loss on crash).
Models are downloaded on first run (dense + sparse, cached locally).

> Sparse vectors use Qdrant's `Modifier.IDF` schema. `build_index.py` detects a
> mismatched collection and takes a snapshot before recreating it, then rebuilds
> (you must re-index in that case).

## 3. Keep the index current (incremental)

`update_index.py` syncs Qdrant to the database without touching the running
app. It fingerprints every published row and, per run, embeds and upserts new
or edited articles and deletes removed ones. It never recreates the collection,
is safe to run while the API is live, and verifies reconciliation at the end
(compares Qdrant point IDs to DB row IDs).

```bash
cd backend
python scripts/update_index.py --init   # seed once, AFTER a full build (no embedding)
python scripts/update_index.py          # scheduled runs
```

State (`last_id` + per-row fingerprints) lives in `backend/data/index_state.json`
(gitignored). When nothing changed, a run is a near-free no-op — no model load.
The script takes its own `flock(2)` on `data/update.lock`, so a cron wrapper
must **not** add another `flock` (they conflict and every run gets skipped).

Every 15 minutes via cron, deprioritized with `nice` (installed by `setup.sh cron`):

```
*/15 * * * * nice -n 15 ~/search-nlp-rag/backend/venv/bin/python \
  ~/search-nlp-rag/backend/scripts/update_index.py \
  >> ~/search-nlp-rag/logs/update_index.log 2>&1
```

## Log management

Logs are capped so they can't fill the disk long-term:

- **Docker** (Qdrant/Redis) run with `--log-driver json-file --log-opt max-size=20m
  --log-opt max-file=3` (set by `setup.sh` via `DOCKER_LOG_OPTS`).
- **App + PM2 logs** (`~/search-nlp-rag/logs/*.log`, `~/.pm2/logs/*.log`) are
  rotated daily by `/etc/logrotate.d/vccircle`: 14 rotations (7 for PM2),
  compressed, with `copytruncate` so open file handles keep writing. Deploy it
  once with:

  ```bash
  sudo install -m 644 /etc/logrotate.d/vccircle  # see repo: deploy/logrotate.conf
  ```

  The config lives in the repo at `deploy/logrotate.conf` for reproducibility.

## 4. Backups and reset

`backup_qdrant.py` snapshots the Qdrant collection and copies the local data
artifacts into `backend/backups/<collection>-<timestamp>/`, keeping the newest
`BACKUP_RETENTION` (default 5) backups:

```bash
cd backend
python scripts/backup_qdrant.py            # snapshot + copy + prune
python scripts/backup_qdrant.py --prune-only
```

`reset_index.py` drops the collection and local artifacts to start from zero.
By default it **blocks** unless a fresh backup succeeded (`--skip-backup`
overrides; `--keep-data` drops the collection only):

```bash
python scripts/reset_index.py               # backup-gated, interactive
python scripts/reset_index.py --yes         # backup-gated, no prompt
python scripts/reset_index.py --keep-data   # drop collection only
```

Backups are local to the host — ship `backend/backups/` (plus
`data/articles.jsonl`) to durable off-server storage for real DR.

## 5. Run the API
```bash
cd backend
./venv/bin/gunicorn -k uvicorn.workers.UvicornWorker --workers 4 \
  --bind 0.0.0.0:8001 --timeout 120 app.main:app
```

| Endpoint | Description |
|---|---|
| `GET /search?q=...&top_k=8` | Hybrid semantic search (no LLM), cached |
| `GET /facets` | Distinct industry/dealtype values for filter autocomplete |
| `POST /api/chat/sessions` | Create a chat conversation |
| `POST /api/chat/sessions/{id}/messages/stream` | SSE-streamed chat turn |
| `GET /health` | Liveness |
| `GET /ready` | Readiness (503 when Qdrant/models unavailable) |
| `GET /live`, `GET /readyz` | Liveness / bare readiness |
| `GET /analytics/dashboard` | Analytics dashboard — a Next.js frontend page (admin-only), fed by the gated `/analytics/summary` + `/analytics/chat` APIs |

```bash
curl "http://localhost:8001/search?q=fintech%20funding&top_k=3"
curl "http://localhost:8001/search?q=funding&industry=Finance,TMT&from_date=2024-01-01"
```

Responses use a slim `SourceSummary` DTO (`id`, `title`, `url`,
`published_date`, `category`, `score`, facet arrays) — article `body`/`summary`
text is used only internally for the LLM context and is never sent to clients.

LLM calls are bounded: `LLM_TIMEOUT_SECONDS`, `LLM_MAX_RETRIES` and
`LLM_RETRY_BACKOFF` control the timeout and exponential backoff; when the model
is unreachable, chat turns return a clean `503` instead of a raw
`500`. Chat requires a valid bearer token (`Authorization: Bearer <token>` from
`/api/auth/login` or `/api/auth/signup`); see `docs/API.md` for the full chat API.

Spend is capped per day via `LLM_DAILY_BUDGET_USD` (0 = disabled): once today's
cumulative LLM cost reaches the cap, chat fails closed (refuses further LLM
calls) rather than racking up unbilled spend.

## 6. Run the frontend

```bash
cd frontend
npm install
npm run dev                  # http://localhost:3000
# production:
npm run build && npm run start
```

Quality gates: `npm run lint` (eslint 9), `npm run typecheck` (`tsc --noEmit`),
`npm run build`. The UI defaults to calling the API **same-origin**
(`location.origin`), which is right when nginx proxies the API paths on the
app's own port. For local dev (`next dev` on :3000 with the API on :8001) set
`NEXT_PUBLIC_API_BASE=http://localhost:8001` — see `.env.local.example`.
`window.API_BASE` before page load overrides anything.

The frontend guards against malformed responses, times out and cancels in-flight
requests (AbortController), sanitizes result URLs (http/https only), maps API
errors to user-friendly messages, and is keyboard/mobile/AT-accessible.
`next.config.ts` sets security headers (CSP, `X-Content-Type-Options`,
`X-Frame-Options`, `Referrer-Policy`).

## Deployment (nginx)

Port map: Qdrant `6333` (internal), API `8001` (internal), Next.js `3000`
(internal), nginx `80` (public). nginx serves the app and proxies the API paths
on the same origin so the UI works with zero CORS setup:

```nginx
server {
    listen 80;
    server_name _;

    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "DENY" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    location /search    { proxy_pass http://127.0.0.1:8001; }
    location /facets    { proxy_pass http://127.0.0.1:8001; }
    location /health    { proxy_pass http://127.0.0.1:8001; }
    location /live      { proxy_pass http://127.0.0.1:8001; }
    location /ready     { proxy_pass http://127.0.0.1:8001; }
    location /api       { proxy_pass http://127.0.0.1:8001; proxy_read_timeout 300s; }
    location /analytics { proxy_pass http://127.0.0.1:8001; }
    location /          { proxy_pass http://127.0.0.1:3000; }
}
```

### Security

Hardening baked into `setup.sh`:

- **Services bound to localhost** — Qdrant and Redis are published as
  `127.0.0.1:PORT:PORT` so they are only reachable from the host (nginx, the
  API), never the internet. If containers were previously created with public
  binds, `setup.sh backend` detects it and recreates them with the local bind
  (Qdrant's data volume is preserved).
- **Host firewall** — enable with UFW (recommended): allow only SSH and HTTP,
  deny the rest:
  ```bash
  sudo ufw default deny incoming
  sudo ufw allow OpenSSH && sudo ufw allow 80/tcp
  sudo ufw --force enable
  ```
  Also restrict the cloud security group (e.g. AWS) to ports 22/80.
- **nginx security headers** — `X-Content-Type-Options: nosniff`,
  `X-Frame-Options: DENY` and `Referrer-Policy: strict-origin-when-cross-origin`
  on every location. CSP is set by the frontend (`middleware.ts`, per-request
  nonce), so it is not duplicated at nginx. Plain HTTP only.
- **Pinned images** — Qdrant/Redis run from pinned, digest-resolvable tags
  (`QDRANT_IMAGE=qdrant/qdrant:v1.19.0@sha256:057e...d1fc`,
  `REDIS_IMAGE=redis:7-alpine`). When overriding Qdrant, keep it >= the version
  that wrote any existing collection — older releases cannot read newer storage
  formats.
- **API note** — CORS is restricted to the origins in `CORS_ORIGINS`
  (localhost dev origins by default; production is same-origin through nginx).
  There is **no auth or rate limiting** on the API. The API is only reachable
  via nginx in the reference deployment; if you expose it directly, add auth and
  rate limits. LLM spend is bounded by `LLM_DAILY_BUDGET_USD` (see
  `app/cost_budget.py`).
- **Health monitoring** — `deploy/healthcheck.sh` probes `/health` (run from
  cron every few minutes), restarts `vccircle-backend` when unhealthy, and logs
  if a restart doesn't recover the app. Logs to `logs/healthcheck.log`.

## Supported settings

All optional (`backend/.env`), see `.env.example` for the full list:

| Variable | Default | Purpose |
|---|---|---|
| `MYSQL_HOST/PORT/USER/PASSWORD/DATABASE/TABLE` | `localhost/3306/root//vccircle/articles` | Source DB (`vcc_frontend`, pk `feid`, `status=1`) |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `QDRANT_COLLECTION` | `vccircle_articles` | Collection name |
| `REDIS_URL` | `redis://localhost:6379/0` | Shared query cache (falls back to in-process cache if Redis is down) |
| `EMBED_MODEL` / `SPARSE_MODEL` | `BAAI/bge-base-en-v1.5` / `Qdrant/bm25` | Dense / sparse embedders (must match index time) |
| `EMBED_DENSE_CHAR_LIMIT` / `EMBED_CHAR_LIMIT` / `BODY_CHAR_LIMIT` | `1500` / `50000` / `50000` | Chars for the dense vector; sparse/lexical vector input; body chars kept in the Qdrant payload |
| `CHAT_BODY_CHAR_LIMIT` | `50000` | Body chars per source fed to the chat LLM (whole stored body by default; lower to cut prompt tokens/cost) |
| `CHAT_MAX_SOURCES` / `CHAT_TOTAL_BODY_CHARS` | `20` / `400000` | Chat context budget: max sources per turn and total body chars across all sources |
| `INDEXER_WORKERS` / `EMBED_BATCH_SIZE` | `2` / `256` | Encode/upsert pipeline depth; embedder batch size (each in-flight batch peaks ~1-2GB on CPU) |
| `EMBED_DEVICE` | `cpu` | `cuda` for a GPU |
| `RERANK_MODEL` / `RERANK_CANDIDATES` | `cross-encoder/ms-marco-MiniLM-L-6-v2` / `12` | Cross-encoder reranker; how many RRF candidates to re-score (12 keeps top-8 quality vs 16, faster on CPU) |
| `RERANK_BACKEND` / `RERANK_ONNX_DIR` | `onnx` / `data/reranker_onnx` | Reranker backend (`onnx` via optimum, `torch` fallback); ONNX export cache dir |
| `TORCH_THREADS` | `2` | Caps torch/onnxruntime threads per worker (also sets OMP/MKL) |
| `ENABLE_QUERY_FIX` / `QUERY_FIX_VOCAB_PATH` | `true` / `data/query_vocab.json.gz` | SymSpell query typo correction (see `app/query_fix.py`); vocab built by `scripts/build_query_vocab.py` |
| `QUERY_FIX_MAX_EDIT` / `QUERY_FIX_MIN_COUNT` / `QUERY_FIX_MIN_TOKEN_LEN` | `2` / `5` / `3` | Max edit distance (short tokens capped at 1); minimum suggestion frequency; minimum token length to consider |
| `ENABLE_DIVERSITY` / `DIVERSITY_LAMBDA` / `DIVERSITY_SIM_THRESHOLD` | `true` / `0.7` / `0.4` | MMR title-diversity reordering of search results (see `app/diversity.py`) |
| `ENABLE_CLICK_BOOST` / `CLICK_BOOST_MIN_CLICKS` / `CLICK_BOOST_MIN_ARTICLE_CLICKS` / `CLICK_BOOST_MIN_SHARE` / `CLICK_BOOST_MULT` | `true` / `5` / `3` / `0.3` / `1.3` | Click-signal score boost on search results (see `app/click_boost.py`) |
| `GEMINI_API_KEY` / `GEMINI_BASE_URL` / `LLM_MODEL` (`GEMINI_MODEL`) | — | Google Gemini (OpenAI-compatible endpoint) for chat |
| `LLM_PRICE_INPUT_PER_1M` / `LLM_PRICE_OUTPUT_PER_1M` / `INR_PER_USD` | `0.25` / `1.50` / `95.60` | USD per 1M input/output tokens (for cost display); USD→INR rate |
| `LLM_DAILY_BUDGET_USD` | `0` (disabled) | Daily LLM spend cap; chat fails closed (refuses LLM calls) once today's cumulative cost reaches this value (see `app/cost_budget.py`) |
| `LLM_TIMEOUT_SECONDS` / `LLM_MAX_RETRIES` / `LLM_RETRY_BACKOFF` | `60` / `2` / `1.0` | LLM per-call timeout, retry count, exponential-backoff base |
| `TOP_K` / `ASK_MIN_SCORE` | `8` / `0.2` | Default result count; chat retrieval threshold |
| `CACHE_TTL_SECONDS` / `CACHE_MAX_SIZE` | `300` / `1000` | Query cache TTL; size of the in-process fallback cache |
| `VECTOR_CACHE_TTL_SECONDS` | `86400` | TTL for the query-time vector cache (query → embedding) |
| `CHAT_DB_PATH` / `CHAT_RETENTION_DAYS` / `CHAT_MAX_HISTORY_TURNS` | `data/chat.db` / `180` / `10` | SQLite chat store; idle-purge window; context turns kept per conversation |
| `RECENCY_STRENGTH` / `RECENCY_DECAY_DAYS` | `0.25` / `90` | Recency-tempered ranking blend |
| `ENABLE_QUERY_EXPANSION` / `ENABLE_ENTITY_BOOST` / `ENABLE_WEAK_FALLBACK` | `true` / `true` / `true` | Query-synonym expansion; entity-mention rerank boost; honest weak-result fallback (see `app/query_expand.py`, `app/rerank_boost.py`, `app/answer_fallback.py`) |
| `ENABLE_BODY_RESCUE` / `BODY_RESCUE_THRESHOLD` | `true` / `0.3` | Chat-only rescue: when the top reranked score is weak, re-score candidates against the body region with the most lexical query-token overlap and keep `max(baseline, body)` so deep-body matches (e.g. retrospectives) can pass `ASK_MIN_SCORE`; one extra cross-encoder pass per candidate, only on weak results |
| `ANALYTICS_REDIS_DB` | `1` | Analytics aggregates live in Redis DB N (cache is DB 0) |
| `AUTH_DB_PATH` / `AUTH_TOKEN_TTL_DAYS` / `AUTH_DEFAULT_ROLE` | `data/auth.db` / `7` / `user` | SQLite auth store; bearer-token expiry; role granted to new public signups |
| `AUTH_SERVICE_TOKEN` | `` (disabled) | Machine-to-machine bypass: a request with this exact `X-Service-Token` header acts as an admin (used by the eval scripts); leave empty to disable |
| `AUTH_ADMIN_EMAIL` / `AUTH_ADMIN_PASSWORD` | `` | Bootstrap admin, created once at startup if absent (never overwrites an existing account) |
| `AUTH_PASSWORD_MIN_LEN` / `AUTH_MAX_EMAIL_LEN` / `AUTH_MAX_NAME_LEN` | `8` / `254` / `60` | Server-side input-validation limits for the auth endpoints |
| `AUTH_SIGNUP_RATE_PER_MIN` / `AUTH_LOGIN_RATE_PER_MIN` / `AUTH_RATE_WINDOW_SECONDS` | `5` / `10` / `60` | Redis-backed per-IP rate limits on public auth endpoints (0 disables) |
| `CORS_ORIGINS` | `http://localhost:3000,http://localhost:8000` | Comma-separated allowed origins for CORS (production is same-origin through nginx) |

## Testing and CI

Backend tests (pytest, fully offline — mocked Qdrant/Redis/MySQL/LLM;
**285 tests, 86% overall coverage**):

```bash
cd backend
pip install -r requirements-dev.txt
python -m pytest tests -q
python -m pytest tests --cov=app --cov-report=term-missing   # coverage report
python -m ruff check app scripts tests
```

Coverage: query-intent/date parsing, facet filter construction, effective
intent, ranking + recency, RAG DTO (no body leak), LLM config wiring, chat store
(CRUD, ownership isolation, retention, token/cost stats) and SSE streaming
(small-talk short-circuit + full-turn deltas, dataviz emission + sanitization),
index fingerprinting/delta/reconciliation, cache TTL and degraded fallback, and
the startup model-load fallbacks. The LLM retry/timeout engine
(`app/llm.py`), the ONNX reranker (`app/reranker.py`), the dense encoder
(`app/encoders.py`), MMR diversity (`app/diversity.py`) and the query fixer
(`app/query_fix.py`) are each covered to **100%** (imports faked, no real model
downloads or inference). `TEST_COVERAGE_GAPS.md` is a live checklist of the
remaining uncovered branches, prioritised error-paths-first (Qdrant/Redis down,
malformed stored rows, LLM budget/retry exhaustion).

`.github/workflows/ci.yml` runs three gates on push/PR to `main`:

1. **backend** — pytest + ruff on Python 3.11
2. **frontend** — eslint, `tsc --noEmit`, production build on Node 22
3. **security** — `pip-audit`, `npm audit --audit-level=high`, gitleaks secret scan

## Eval scripts (repeatable quality checks)

Run on a box with the backend serving on localhost:8001 and MySQL/Qdrant
reachable from `backend/.env` (venv python, from `backend/`). The scripts
authenticate with `AUTH_SERVICE_TOKEN` from `backend/.env` (auto-loaded), so no
manual token setup is needed:

- `./venv/bin/python scripts/deep_body_eval.py` — whole-body indexing and
  body-grounded chat: (1) coverage of terms that live only in the deep body
  (>6000 chars) inside the stored sparse vectors, (2) title+deep-word search
  top-8 hits, (3) a chat query answered purely from an article body whose
  summary is empty. Deterministic seed.
- `./venv/bin/python scripts/chat_quality_test.py` — golden-set chat eval
  (event-year date suppression, plain-year date filters, weak-fallback niche,
  normal and entity queries) verifying key terms, expected source years, and —
  for the dataviz-intent queries — that the answer carries a valid
  ```` ```dataviz ```` JSON block (same contract `app.chat.parse_dataviz`
  enforces). Makes real (billed) LLM calls.
- `./venv/bin/python scripts/search_quality_test.py [top_k]` — golden-set
  search eval over the live API (default `top_k=8`), checking that the right
  articles surface for normal, entity, date-filtered and niche queries.
- `./venv/bin/python scripts/rerank_bench.py` — reranker latency/quality
  benchmark (ONNX vs torch backends) over representative queries.
- `./venv/bin/python scripts/load_test.py` — concurrency/throughput harness
  against the live API (workers, duration and query file configurable).

