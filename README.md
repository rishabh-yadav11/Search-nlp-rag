# VCCircle New Search

Hybrid retrieval + RAG search over the VCCircle article corpus. Articles are
indexed into Qdrant with dense (semantic) and sparse (BM25) vectors fused with
Reciprocal Rank Fusion (RRF). A FastAPI service queries the index and optionally
synthesizes cited answers with an LLM (Groq). A Next.js single-page UI ties it
together.

```
MySQL ──fetch_data──▶ articles.jsonl ──build_index──▶ Qdrant (dense + sparse)
   │                                                         ▲
   └─────────update_index (incremental new/edit/delete)──────┘
                                                               │
Browsers ◀── nginx ───▶ Next.js app ◀──(same origin)──▶ FastAPI ──▶ Qdrant
                         /search, /ask, /health,               │
                         /live, /ready                       Groq (for /ask)
                                                              Redis (cache)
```

## Repository layout

```
backend/
  app/
    main.py            FastAPI app: /search, /ask (RAG), /health
    health.py          /live, /ready, /readyz dependency checks
    llm.py             LLM call with timeout, retries, backoff
    config.py          env-driven settings
    query_intent.py    year/top-N intent parsing + Flashback rewriting
    index_text.py      shared text composition + date normalization
    redis_cache.py     Redis-backed cache with in-process fallback
  scripts/
    fetch_data.py      MySQL -> data/articles.jsonl (paginated, resumable)
    build_index.py     articles.jsonl -> Qdrant embeddings (checkpointed)
    update_index.py    incremental MySQL->Qdrant sync (new/edit/delete)
    backup_qdrant.py   Qdrant snapshot + local artifact backups (retention)
    qdrant_backup.py   shared backup helpers
    reset_index.py     drop the index + data files (backup-gated)
  tests/               pytest suite (offline, mocked deps)
  requirements.txt
  requirements-dev.txt lint/test tooling
  .env.example         configuration template
frontend/              Next.js app (App Router + TypeScript)
  app/page.tsx         search/ask UI (timeouts, validation, a11y)
  app/globals.css
  next.config.ts       security headers (CSP, nosniff, etc.)
  eslint.config.mjs    flat config for eslint 9
setup.sh               one-command deploy (deps, services, index, nginx, cron)
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
- **Faceted filtering** — `/search` and `/ask` accept optional `industry`,
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
- **Ask mode (RAG)** — retrieves articles above `ASK_MIN_SCORE`, packs them
  (title, date, authors/industry/dealtype, summary and a body excerpt) into a
  numbered context, and asks Groq (`llama-3.3-70b-versatile`) for an answer
  with inline `[n]` citations. If nothing clears the threshold it says so
  instead of guessing.
- **Caching** — Redis-backed JSON cache shared across workers (falls back to
  an in-process cache if Redis is down) for both `/search` and `/ask`, keyed
  by effective query + top_k + facets.

### Health endpoints

| Endpoint | Purpose | Fails on |
|---|---|---|
| `/health` | Liveness (always 200 if the process is up) | — |
| `/live` | Liveness alias | — |
| `/ready` | Readiness, JSON report | Qdrant down or models not loaded → `503` |
| `/readyz` | Readiness, bare status code | same as `/ready` |

`/ready` checks the Qdrant collection, model/reranker loading, and reports Redis
and Groq status non-fatally (Redis failures degrade to the in-process cache, so
they don't flip readiness).

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
cp .env.example .env        # fill in MySQL creds + GROQ_API_KEY
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
| `GET /ask?q=...&top_k=8` | Search + cited LLM answer, cached |
| `GET /health` | Liveness |
| `GET /ready` | Readiness (503 when Qdrant/models unavailable) |
| `GET /live`, `GET /readyz` | Liveness / bare readiness |

```bash
curl "http://localhost:8001/search?q=fintech%20funding&top_k=3"
curl "http://localhost:8001/search?q=funding&industry=Finance,TMT&from_date=2024-01-01"
curl "http://localhost:8001/ask?q=who%20is%20investing%20in%20fintech&top_k=5"
```

Responses use a slim `SourceSummary` DTO (`id`, `title`, `url`,
`published_date`, `category`, `score`, facet arrays) — article `body`/`summary`
text is used only internally for the LLM context and is never sent to clients.

LLM calls are bounded: `LLM_TIMEOUT_SECONDS`, `LLM_MAX_RETRIES` and
`LLM_RETRY_BACKOFF` control the timeout and exponential backoff; when the model
is unreachable, `/ask` returns a clean `503` instead of a raw `500`.

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

    location /search { proxy_pass http://127.0.0.1:8001; }
    location /health { proxy_pass http://127.0.0.1:8001; }
    location /live   { proxy_pass http://127.0.0.1:8001; }
    location /ready  { proxy_pass http://127.0.0.1:8001; }
    location /ask    { proxy_pass http://127.0.0.1:8001; proxy_read_timeout 300s; }
    location /       { proxy_pass http://127.0.0.1:3000; }
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
  on every location. CSP is set by the frontend (`next.config` headers), so it
  is not duplicated at nginx. Plain HTTP only.
- **Pinned images** — Qdrant/Redis run from pinned, digest-resolvable tags
  (`QDRANT_IMAGE=qdrant/qdrant:v1.19.0@sha256:057e...d1fc`,
  `REDIS_IMAGE=redis:7-alpine`). When overriding Qdrant, keep it >= the version
  that wrote any existing collection — older releases cannot read newer storage
  formats.
- **API note** — CORS is currently wide open (`allow_origins=["*"]`) for POC
  convenience and there is **no auth or rate limiting**. The API is only reachable
  via nginx in the reference deployment; if you expose it directly, add auth,
  rate limits, and restrict CORS to the real frontend origin.

## Supported settings

All optional (`backend/.env`), see `.env.example` for the full list:

| Variable | Default | Purpose |
|---|---|---|
| `MYSQL_HOST/PORT/USER/PASSWORD/DATABASE/TABLE` | `localhost/3306/root//vccircle/articles` | Source DB (`vcc_frontend`, pk `feid`, `status=1`) |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `QDRANT_COLLECTION` | `vccircle_articles` | Collection name |
| `REDIS_URL` | `redis://localhost:6379/0` | Shared query cache (falls back to in-process cache if Redis is down) |
| `EMBED_MODEL` / `SPARSE_MODEL` | `BAAI/bge-base-en-v1.5` / `Qdrant/bm25` | Dense / sparse embedders (must match index time) |
| `EMBED_DENSE_CHAR_LIMIT` / `EMBED_CHAR_LIMIT` / `BODY_CHAR_LIMIT` | `1500` / `6000` / `6000` | Chars for the dense vector; sparse/lexical vector; body chars kept in the payload |
| `INDEXER_WORKERS` / `EMBED_BATCH_SIZE` | `2` / `256` | Encode/upsert pipeline depth; embedder batch size (each in-flight batch peaks ~1-2GB on CPU) |
| `EMBED_DEVICE` | `cpu` | `cuda` for a GPU |
| `RERANK_MODEL` / `RERANK_CANDIDATES` | `cross-encoder/ms-marco-MiniLM-L-6-v2` / `16` | Cross-encoder reranker; how many RRF candidates to re-score |
| `GROQ_API_KEY` / `GROQ_BASE_URL` / `LLM_MODEL` | — | Groq-compatible endpoint for `/ask` |
| `LLM_TIMEOUT_SECONDS` / `LLM_MAX_RETRIES` / `LLM_RETRY_BACKOFF` | `60` / `2` / `1.0` | LLM per-call timeout, retry count, exponential-backoff base |
| `TOP_K` / `ASK_MIN_SCORE` | `8` / `0.2` | Default result count; `/ask` retrieval threshold |
| `CACHE_TTL_SECONDS` / `CACHE_MAX_SIZE` | `120` / `1000` | Query cache TTL; size of the in-process fallback cache |
| `RECENCY_STRENGTH` / `RECENCY_DECAY_DAYS` | `0.25` / `90` | Recency-tempered ranking blend |
| `ENABLE_QUERY_EXPANSION` / `ENABLE_ENTITY_BOOST` / `ENABLE_WEAK_FALLBACK` | `true` / `true` / `true` | Query-synonym expansion; entity-mention rerank boost; honest weak-result fallback (see `app/query_expand.py`, `app/rerank_boost.py`, `app/answer_fallback.py`) |

## Testing and CI

Backend tests (pytest, fully offline — mocked Qdrant/Redis/MySQL/LLM):

```bash
cd backend
pip install -r requirements-dev.txt
python -m pytest tests -q
python -m ruff check app scripts tests
```

Coverage: query-intent/date parsing, facet filter construction, effective
intent, ranking + recency, RAG DTO (no body leak), LLM config wiring, index
fingerprinting/delta/reconciliation, cache TTL and degraded fallback.

`.github/workflows/ci.yml` runs three gates on push/PR to `main`:

1. **backend** — pytest + ruff on Python 3.11
2. **frontend** — eslint, `tsc --noEmit`, production build on Node 22
3. **security** — `pip-audit`, `npm audit --audit-level=high`, gitleaks secret scan

## Production notes (POC shortcuts to fix before real prod)

1. **Embeddings and the reranker run on CPU.** Query-time embedding and
   cross-encoder re-scoring are offloaded off the event loop, so the API stays
   responsive, but latency is higher than a GPU-backed embedder. Set
   `EMBED_DEVICE=cuda` for lower latency.
2. **Cache is Redis-backed** and shared across gunicorn workers; if Redis is
   down it silently degrades to a per-worker in-process cache (which no longer
   benefits the other workers until Redis returns).
3. **No auth/rate limiting** on the API — add before exposing beyond
   nginx/localhost.
4. **`published_date` payload index** assumes MySQL returns a parseable
   date/datetime string; adjust the payload schema if your column type differs.
5. **Backups are host-local** — schedule off-server transfer of
   `backend/backups/` and `backend/data/articles.jsonl` for disaster recovery.
