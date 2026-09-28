# VCCircle News Search

![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12-blue)
![Node](https://img.shields.io/badge/node-18%2B-339933)
![FastAPI](https://img.shields.io/badge/backend-FastAPI-009688)
![Next.js](https://img.shields.io/badge/frontend-Next.js-000000)
![Qdrant](https://img.shields.io/badge/vector%20db-Qdrant-DC244C)
![License](https://img.shields.io/badge/license-unspecified-lightgrey)
![CI](https://img.shields.io/badge/CI-backend%20%7C%20frontend%20%7C%20security-brightgreen)

I built this as a hybrid retrieval + RAG search system over the VCCircle article corpus, with a ChatGPT-style chat assistant on top. I index articles into Qdrant using both dense (semantic) and sparse (BM25) vectors, fuse them with Reciprocal Rank Fusion (RRF), and serve everything through a FastAPI backend. An LLM (Google Gemini, via an OpenAI-compatible endpoint) synthesizes cited answers, and I store per-user chat conversations in SQLite. The frontend is a Next.js app with a search page and a `/chat` page.

## Table of Contents

- [Architecture](#architecture)
- [Repository Layout](#repository-layout)
- [How It Works](#how-it-works)
- [Health Endpoints](#health-endpoints)
- [Prerequisites](#prerequisites)
- [One-Command Deployment](#one-command-deployment-recommended)
- [1. Backend Setup](#1-backend-setup)
- [2. Build the Index](#2-build-the-index-run-once)
- [3. Keep the Index Current](#3-keep-the-index-current-incremental)
- [Log Management](#log-management)
- [4. Backups and Reset](#4-backups-and-reset)
- [5. Run the API](#5-run-the-api)
- [6. Run the Frontend](#6-run-the-frontend)
- [Deployment (nginx)](#deployment-nginx)
- [TLS (HTTPS)](#tls-https)
- [Security](#security)
- [Supported Settings](#supported-settings)
- [Testing and CI](#testing-and-ci)

## Architecture

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

## Repository Layout

```
backend/
  app/
    main.py            FastAPI app: /search, /chat, /health, /analytics
    health.py          /health, /live, /ready, /ready/deep, /readyz dependency checks
    llm.py             LLM call with timeout, retries, backoff, token-cost calc
    chat.py            per-user chat store (SQLite) + /api/chat router
    analytics.py       Redis-backed search/click analytics aggregates
    config.py          env-driven settings
    query_intent.py    year/top-N intent parsing + Flashback rewriting
    query_expand.py    deterministic synonym query expansion
    rerank_boost.py    entity-mention score boost on reranked results
    answer_fallback.py weak-result notes + honest chat fallback replies
    index_text.py      shared text composition + date normalization
    redis_cache.py     Redis-backed cache with in-process fallback
  scripts/
    fetch_data.py      MySQL -> data/articles.jsonl (paginated, resumable)
    build_index.py     articles.jsonl -> Qdrant embeddings (checkpointed)
    update_index.py    incremental MySQL->Qdrant sync (new/edit/delete)
    backfill_summary.py  one-off summary payload backfill
    backup_qdrant.py   Qdrant snapshot + local artifact backups (retention)
    qdrant_backup.py   shared backup helpers
    reset_index.py     drop the index + data files (backup-gated)
  tests/               pytest suite (offline, mocked deps)
  requirements.txt
  requirements-dev.txt lint/test tooling
  .env.example         configuration template
frontend/              Next.js app (App Router + TypeScript)
  app/page.tsx         search UI (timeouts, validation, a11y)
  app/chat/page.tsx    ChatGPT-style chat UI (SSE streaming)
  app/globals.css
  middleware.ts        CSP nonce header
  next.config.ts       security headers (CSP, nosniff, etc.)
  eslint.config.mjs    flat config for eslint 9
setup.sh               one-command deploy (deps, services, index, nginx, cron)
.github/workflows/ci.yml   backend + frontend + security gates
```

## How It Works

- **Hybrid retrieval** — I embed every article twice at index time: `BAAI/bge-base-en-v1.5` (dense, cosine, 768-dim) and `Qdrant/bm25` sparse vectors with IDF. At query time I search both in a single Qdrant prefetch and fuse them with RRF. The **dense** vector is `title + authors + industry + dealtype + summary` (metadata first, no body, short and fast to encode on CPU); the **sparse (BM25)** vector adds the full `body`, so keyword matches inside article bodies stay searchable at lexical cost.
- **Faceted filtering** — `/search` accepts optional `industry`, `dealtype`, `author`, and `from_date`/`to_date` params, applied as a Qdrant filter to both prefetches. Any value can be comma-separated for multi-select. Cache keys include the filters so distinct queries don't collide.
- **Query intent** — I parse relative/absolute years ("last year", "2025", spans) into automatic date filters, and rewrite "top/best N X in Y" to "Flashback Y X" for year-review retrieval with a bumped `top_k`. Explicit user-supplied dates always win.
- **Reranking** — RRF candidates get re-scored with a cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`) over `title + summary`; the final `top_k` reflects true relevance, and the score I show the user is that reranked score (0-1).
- **Result ordering** — recency-tempered relevance first: blended score desc (`score * (1 - RECENCY_STRENGTH * (1 - exp(-age_days / RECENCY_DECAY_DAYS)))`), then `published_date` desc as tie-break (missing dates last). A query that itself asks for fresh news ('latest'/'recent'/'fresh') swaps in the much heavier `RECENCY_BOOST_STRENGTH` / `RECENCY_BOOST_DECAY_DAYS` pair, so old evergreen articles drop below recent ones instead of surfacing on relevance alone; hard windows like 'this week' are filtered, not boosted.
- **Chat (conversations)** — a ChatGPT-style UI at `/chat`. Each device gets an anonymous `X-User-Id` (a localStorage UUID), and I store conversations per user in SQLite (`backend/data/chat.db`, WAL mode) so they survive restarts and stay shared across gunicorn workers. Turns reuse the same retrieval/rerank/fallback pipeline as search but build a conversation-aware prompt; the answer streams token-by-token over SSE (`POST .../messages/stream`), and I persist the assistant message, sources, tokens, cost, and latency. Conversations idle for `CHAT_RETENTION_DAYS` (180) get purged daily. See `docs/API.md`.
- **Caching** — a Redis-backed JSON cache shared across workers (falls back to an in-process cache if Redis is down) for `/search`, keyed by effective query + top_k + facets. I also cache the retrieval/rerank step (`retrieve_and_rerank`) on the same key space (query + filter + top_k), so repeated searches and every chat turn that re-asks the same question skip embedding + rerank entirely.

### Health Endpoints

| Endpoint  | Purpose                                    | Fails on                                 |
| --------- | ------------------------------------------ | ----------------------------------------- |
| `/health` | Liveness (always 200 if the process is up) | —                                         |
| `/live`   | Liveness alias                             | —                                         |
| `/ready`  | Readiness, JSON report                     | Qdrant down, models not loaded, or no usable `GEMINI_API_KEY` → `503` |
| `/readyz` | Readiness, bare status code                | same as `/ready`                          |
| `/ready/deep` | Readiness, uncached and unrated         | same as `/ready`; loopback callers only   |

`/ready` checks the Qdrant collection, model/reranker loading and the Gemini
key, and reports Redis non-fatally (a Redis failure degrades to the in-process
cache, so it does not flip readiness). An **absent or placeholder**
`GEMINI_API_KEY` is fatal to readiness: chat answers every question from the
canned fallback, so the deployment is not fit to serve. `checks.llm.reason` is
`ok`, `missing`, `placeholder` or `malformed` and never contains the key.

`/health` and `/live` check nothing at all — that is what makes them valid
liveness probes, and also why they must never gate a deploy or an alert. Anything
needing a real answer uses `/ready` (load balancer, cached and rate limited) or
`/ready/deep` (the deploy gate and the cron watchdog, uncached and unrated so
they cannot read a stale verdict or be throttled into a false outage).
`/ready/deep` answers only a direct loopback request with no
`X-Forwarded-For`, and the vhost below additionally returns `404` for it, so the
unrated, uncached probe has two independent layers in front of it rather than
depending on one line of application check.

## Prerequisites

- Python 3.11–3.12 (warned-but-tolerated on 3.13/3.14; set `ALLOW_UNSUPPORTED_PY=1` to silence)
- Node.js 18+ (22 recommended for the frontend)
- Node >= 22.6 to run the backend test suite's cross-language dataviz
  contract test (`backend/tests/test_dataviz_contract.py`): it imports the
  shipped `frontend/app/chat/datavizContract.ts` directly, which needs the
  built-in TypeScript support in node. Without it that test skips locally
  and FAILS under CI, so the browser-side validator can never go untested
  silently.
- Docker (for Qdrant and Redis)
- MySQL source database

## One-Command Deployment (recommended)

I wrote `setup.sh` to provision everything in stages. Run `./setup.sh all`, or pick individual stages:

```bash
./setup.sh backend      # python check, venv + deps, Qdrant/Redis (bound to 127.0.0.1)
./setup.sh index        # fetch_data -> build_index -> seed incremental state
./setup.sh frontend     # npm ci + production build
./setup.sh services     # pm2 start gunicorn (API) + next (frontend)
./setup.sh pm2-startup  # systemd unit so services restore on reboot
./setup.sh cron         # 15-min incremental sync
./setup.sh nginx        # reverse proxy + security headers on :80
./setup.sh tls          # HTTPS: certbot + :443 + http->https redirect (needs a domain; see [TLS](#tls-https))
./setup.sh all          # deps backend index frontend services pm2-startup cron nginx
```

`tls` is deliberately not part of `./setup.sh all`: it needs a domain name, a contact address and network access that an unattended bootstrap must not require. Run it once, on purpose.

Environment overrides: `QDRANT_PORT`, `REDIS_PORT`, `API_PORT`, `NEXT_PORT`, `PUBLIC_PORT`, `GUNICORN_WORKERS`, `PUBLIC_BASE_URL`, `QDRANT_IMAGE`, `REDIS_IMAGE`, `ALLOW_UNSUPPORTED_PY`, plus the TLS knobs `NGINX_TLS` (`auto`/`on`/`off`), `LE_DOMAIN`, `LE_EMAIL`, `LE_ROOT`, `CERTBOT_WEBROOT`, `NGINX_CONF`, `NGINX_LINK`. pm2 process tuning: `API_MAX_MEMORY` (5G), `FRONTEND_MAX_MEMORY` (1G), `API_MAX_RESTARTS` (10), `RESTART_BACKOFF_MS` (100) — these must stay equal to `ecosystem.config.js`, and `backend/tests/test_deploy_config.py` fails the build if the two process definitions drift apart.

If you'd rather run pieces manually, keep reading.

## 1. Backend Setup

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

## 2. Build the Index (run once)

```bash
cd backend
python scripts/fetch_data.py     # MySQL -> data/articles.jsonl
python scripts/build_index.py    # embed + upsert into Qdrant
```

Both scripts are safe to interrupt and re-run: `fetch_data.py` resumes from the max id already written; `build_index.py` resumes from a line-number checkpoint and only advances it after Qdrant acknowledges the upsert (no silent data loss on crash). Models download on first run (dense + sparse, cached locally).

> Sparse vectors use Qdrant's `Modifier.IDF` schema. `build_index.py` detects a mismatched collection, takes a snapshot before recreating it, then rebuilds (you'll need to re-index in that case).

## 3. Keep the Index Current (incremental)

`update_index.py` syncs Qdrant to the database without touching the running app. It fingerprints every published row and, per run, embeds and upserts new or edited articles and deletes removed ones. It never recreates the collection, is safe to run while the API is live, and verifies reconciliation at the end (compares Qdrant point IDs to DB row IDs).

```bash
cd backend
python scripts/update_index.py --init   # seed once, AFTER a full build (no embedding)
python scripts/update_index.py          # scheduled runs
```

State (`last_id` + per-row fingerprints) lives in `backend/data/index_state.json` (gitignored). When nothing changed, a run is a near-free no-op — no model load. The script takes its own `flock(2)` on `data/update.lock`, so a cron wrapper must **not** add another `flock` (they'd conflict and every run would get skipped).

Every 15 minutes via cron, deprioritized with `nice` (installed by `setup.sh cron`):

```bash
*/15 * * * * nice -n 15 ~/search-nlp-rag/backend/venv/bin/python \
  ~/search-nlp-rag/backend/scripts/update_index.py \
  >> ~/search-nlp-rag/logs/update_index.log 2>&1
```

## Log Management

I capped logs so they can't fill the disk long-term:

- **Docker** (Qdrant/Redis) runs with `--log-driver json-file --log-opt max-size=20m --log-opt max-file=3` (set by `setup.sh` via `DOCKER_LOG_OPTS`).
- **App + PM2 logs** (`<repo>/logs/*.log`, `~/.pm2/logs/*.log`) are rotated daily by `/etc/logrotate.d/vccircle`: 14 rotations (7 for PM2), compressed, with `copytruncate` so open file handles keep writing. The shipped `deploy/logrotate.conf` carries `/path/to/...` placeholders — logrotate does no variable interpolation, so substitute them before installing. Run this from the **repo root** and write to a copy, so the tracked file stays clean and the substitution can't pick up a `backend/` subdirectory:

```bash
cd /path/to/search-nlp-rag            # the repo root
sed "s|/path/to/search-nlp-rag|$PWD|g; s|/path/to/.pm2|$HOME/.pm2|g" \
    deploy/logrotate.conf > /tmp/vccircle-logrotate
sudo install -m 644 /tmp/vccircle-logrotate /etc/logrotate.d/vccircle
```

The `su` directive is commented out by default (logrotate then runs as root, which is correct whenever the logs are root-owned); uncomment and set it to your own `<user> <group>` otherwise. The placeholder paths must be substituted — left in place, `missingok` makes logrotate silently rotate nothing.

The config lives in the repo at `deploy/logrotate.conf` for reproducibility.

### Data locations

`CHAT_DB_PATH`, `AUTH_DB_PATH`, `QUERY_FIX_VOCAB_PATH` and `RERANK_ONNX_DIR` are resolved to **absolute** paths at startup. A relative value (what `.env.example` ships) resolves against the backend directory, never against the process working directory, so starting the app from anywhere opens the same database. At boot the API logs the three locations it writes to and refuses to start if one cannot be created or written, rather than silently opening a new empty database and appearing to have lost all history.

## 4. Backups and Reset

`backup_qdrant.py` snapshots the Qdrant collection and copies the local data artifacts into `backend/backups/<collection>-<timestamp>/`, keeping the newest `BACKUP_RETENTION` (default 5) backups:

```bash
cd backend
python scripts/backup_qdrant.py            # snapshot + copy + prune
python scripts/backup_qdrant.py --prune-only
```

`backup_qdrant.py` exits non-zero unless a verified local snapshot archive was written, so a cron job or wrapper can detect a backup that wrote nothing.

`reset_index.py` drops the collection and local artifacts to start from zero. By default it **blocks** unless a fresh backup succeeded, and "succeeded" means a snapshot archive was downloaded to `backend/backups/` and verified as a readable tar — a snapshot that exists only inside the Qdrant container does not count, because a container recreate or `docker rm` destroys it (`--skip-backup` overrides; `--keep-data` drops the collection only). It exits non-zero on any abort, so a reset that did not run is never mistaken for one that did:

```bash
python scripts/reset_index.py               # backup-gated, interactive
python scripts/reset_index.py --yes         # backup-gated, no prompt
python scripts/reset_index.py --keep-data   # drop collection only
```

Backups are local to the host — ship `backend/backups/` (plus `data/articles.jsonl`) to durable off-server storage for real DR.

## 5. Run the API

```bash
cd backend
./venv/bin/gunicorn -k uvicorn.workers.UvicornWorker --workers 4 \
  --bind 127.0.0.1:8001 --timeout 120 app.main:app
```

| Endpoint                                       | Description                                                 |
| ----------------------------------------------- | ------------------------------------------------------------ |
| `GET /search?q=...&top_k=8`                     | Hybrid semantic search (no LLM), cached                      |
| `GET /facets`                                   | Distinct industry/dealtype values for filter autocomplete    |
| `POST /api/chat/sessions`                       | Create a chat conversation                                   |
| `POST /api/chat/sessions/{id}/messages/stream`  | SSE-streamed chat turn                                       |
| `GET /health`                                   | Liveness (checks nothing; never gates anything)               |
| `GET /ready`                                    | Readiness (503 on Qdrant/models/Gemini-key failure)            |
| `GET /live`, `GET /readyz`                      | Liveness alias / bare readiness                                |
| `GET /ready/deep`                               | Readiness for local monitoring: no cache, no rate limit, `403` from non-loopback |

```bash
curl "http://localhost:8001/search?q=fintech%20funding&top_k=3"
curl "http://localhost:8001/search?q=funding&industry=Finance,TMT&from_date=2024-01-01"
```

Responses use a slim `SourceSummary` DTO (`id`, `title`, `url`, `published_date`, `category`, `score`, facet arrays) — article `body`/`summary` text is used only internally for the LLM context and is never sent to clients.

LLM calls are bounded: `LLM_TIMEOUT_SECONDS`, `LLM_MAX_RETRIES`, and `LLM_RETRY_BACKOFF` control the timeout and exponential backoff; when the model is unreachable, chat turns return a clean `503` instead of a raw `500`. Chat conversations require the `X-User-Id` header (min 8 chars) — see `docs/API.md` for the full chat API.

Spend is capped per day via `LLM_DAILY_BUDGET_USD` (default `5.0`; `0` deliberately disables the cap): each turn reserves its share of the cap atomically *before* the billed LLM call, so concurrent turns contend for the same budget rather than all passing a stale read, and once today's cumulative LLM cost (plus every in-flight reservation) reaches the cap, chat fails closed (refuses further LLM calls) rather than racking up unbilled spend.

## 6. Run the Frontend

```bash
cd frontend
npm install
npm run dev                  # http://localhost:3000
# production:
npm run build && npm run start
```

Quality gates: `npm run lint` (eslint 9), `npm run typecheck` (`tsc --noEmit`), `npm run build`. The UI defaults to calling the API **same-origin** (`location.origin`), which is right when nginx proxies the API paths on the app's own port. For local dev (`next dev` on :3000 with the API on :8001), set `NEXT_PUBLIC_API_BASE=http://localhost:8001` — see `.env.local.example`. `window.API_BASE` before page load overrides anything.

The frontend guards against malformed responses, times out and cancels in-flight requests (AbortController), sanitizes result URLs (http/https only), maps API errors to user-friendly messages, and is keyboard/mobile/AT-accessible. `next.config.ts` sets security headers (CSP, `X-Content-Type-Options`, `X-Frame-Options`, `Referrer-Policy`).

## Deployment (nginx)

Port map: Qdrant `6333` (internal), API `8001` (internal), Next.js `3000` (internal), nginx `80` (public, plus `443` once TLS is enabled). nginx serves the app and proxies the API paths on the same origin so the UI works with zero CORS setup. `setup.sh nginx` writes this file and is the source of truth — the excerpt below is the plain-HTTP shape, not something to hand-maintain:

Both internal services bind `127.0.0.1`, so nginx on `:80` is the only way in: `next start` and gunicorn are told their loopback address explicitly at every startup path (`setup.sh`, `ecosystem.config.js`, `npm start`) rather than relying on a default, because each of those defaults to all interfaces. `next dev` is a local development server and deliberately keeps the wildcard so you can reach it from a phone on the LAN.

```nginx
server {
    listen 80;
    server_name _;

    add_header X-Content-Type-Options "nosniff" always;
    add_header X-Frame-Options "DENY" always;
    add_header Referrer-Policy "strict-origin-when-cross-origin" always;

    # Always served over plain HTTP, in both modes: Let's Encrypt validates
    # over http, so redirecting the challenge away would break renewal. The
    # "^~" makes this win over the catch-all "location /" below.
    location ^~ /.well-known/acme-challenge/ {
        root /var/www/certbot;
        default_type text/plain;
    }

    # The per-IP rate limiter on /search, /facets, /analytics/click and /ready
    # keys on the client IP these headers carry. Without them every proxied
    # request looks like 127.0.0.1 and the whole site shares one rate-limit bucket.
    # These headers are trusted by default here: with the peer being loopback
    # the API reads the forwarded client IP (AUTH_TRUST_X_FORWARDED_FOR=auto,
    # the shipped default). A client hitting :8001 directly is its own
    # non-loopback peer, so the same header is ignored for it and cannot be
    # used to dodge a rate limit.
    location /search {
        proxy_pass http://127.0.0.1:8001;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
    location /facets {
        proxy_pass http://127.0.0.1:8001;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
    location /health    { proxy_pass http://127.0.0.1:8001; }
    location /live      { proxy_pass http://127.0.0.1:8001; }
    location /ready {
        proxy_pass http://127.0.0.1:8001;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
    location /readyz {
        proxy_pass http://127.0.0.1:8001;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
    # `location /ready` is a PREFIX match, so a public GET /ready/deep would
    # otherwise be proxied here and refused only by the API's own host-local
    # check. /ready/deep is uncached and unrated (it is the watchdog's probe), so
    # it gets a second, independent layer and simply does not exist on the public
    # surface. setup.sh and deploy/healthcheck.sh reach it on 127.0.0.1 directly.
    location /ready/deep { return 404; }
    location /api {
        proxy_pass http://127.0.0.1:8001;
        proxy_read_timeout 300s;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
    location /recommend/ {
        proxy_pass http://127.0.0.1:8001;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
    location /analytics/click {
        proxy_pass http://127.0.0.1:8001;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
    location /analytics/summary { proxy_pass http://127.0.0.1:8001; }
    location /analytics/chat    { proxy_pass http://127.0.0.1:8001; }
    location /analytics { proxy_pass http://127.0.0.1:3000; }
    location /          {
        proxy_pass http://127.0.0.1:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### TLS (HTTPS)

Out of the box the site is served over plain HTTP on port 80. Passwords, bearer tokens, the `X-Service-Token` and chat content all cross the wire in cleartext. The `Strict-Transport-Security` header the frontend sets (`next.config.ts`: `max-age=63072000; includeSubDomains; preload`) is ignored by browsers unless it arrives over https, so until TLS is on that header does nothing at all.

`./setup.sh tls` enables it. It requires:

- **A domain name** whose `A`/`AAAA` record points at this host. Let's Encrypt issues certificates for domain names, never for a bare IP, so this cannot be enabled on a host that is reached by IP alone.
- **Ports 80 and 443 open** to the internet (host firewall *and* cloud security group). Port 80 must stay open even after TLS is on — issuance and renewal both validate over plain HTTP.
- **certbot** on the host: `sudo apt-get install -y certbot`.
- **An ACME contact address** — expiry warnings go there.

```bash
LE_DOMAIN=search.example.com LE_EMAIL=you@example.com ./setup.sh tls
```

The stage is written to be safe to re-run and safe to interrupt:

- It makes sure the challenge is servable **before** asking certbot for anything, because the token has to be reachable at `http://<domain>/.well-known/acme-challenge/<token>` for validation to succeed. On a host whose nginx config predates that location, skipping this step makes the token fall through to Next.js, 404, and issuance fail on the very first run. Port 80 serves the challenge in **both** modes, so this step only drops to plain HTTP when there is no usable certificate yet; with one already in place the site keeps serving HTTPS and only the challenge location is added, which is why `./setup.sh tls` is safe to re-run on a live encrypted host.

- It issues with certbot's **webroot** plugin against `CERTBOT_WEBROOT` (default `/var/www/certbot`) and never `--standalone`. `--standalone` needs port 80 to be free, so on a live site it would either fail outright or force nginx to stop and take the site down.
- `--keep-until-expiring` makes a re-run a no-op instead of consuming Let's Encrypt's rate limits.
- The config is installed only after `nginx -t` accepts it; if nginx rejects it, the previous config is restored. nginx is then **reloaded**, not restarted, so in-flight requests survive.
- If certbot fails, the stage exits non-zero, reports the posture that is actually still serving (HTTPS if a usable certificate was already there, plain HTTP if not), and names the likely cause (DNS not pointing here, or port 80 closed) and where certbot's own log is. A re-run that fails to renew never costs the site the HTTPS it already had.
- nginx is never handed a config that names a certificate pair that is not there: the `443` server block is rendered only when `/etc/letsencrypt/live/<domain>/fullchain.pem` *and* `privkey.pem` are both present and non-empty. Whether nginx can actually *read* them is deliberately not decided here — that check runs as the operator while `nginx -t` runs as root, and certbot writes `privkey.pem` `0600 root:root`, so testing readability would refuse a config nginx loads perfectly well. `nginx -t` is the authority, and it is the gate that rolls back.

Once it runs, port 80 keeps serving `/.well-known/acme-challenge/` and redirects everything else to `https://$host$request_uri`, and a matching `listen 443 ssl http2` server (TLSv1.2/1.3) is added. Both servers share one location body, so they cannot drift apart.

**Renewal.** certbot renews roughly 30 days before expiry. The stage installs a `--deploy-hook` that reloads nginx, and uses the `certbot.timer` systemd unit when it is enabled, otherwise a daily `certbot renew` crontab line. Verify the whole path without touching real certificates:

```bash
sudo certbot renew --dry-run
```

**Rollback to plain HTTP** — one flag, no other moving parts:

```bash
NGINX_TLS=off ./setup.sh nginx
```

**This rollback is not fully reversible in browsers, and that is a property of the 301, not a bug in the flag.** Port 80 answers the redirect with a permanent `301`, which is the right code for a site that is meant to stay on TLS — it is cacheable, so the redirect costs no round trip on every subsequent request. The cost is that clients cache it hard: Chrome honours a 301 for up to ~109 days, and Firefox and Safari keep their own long-lived copies. So after rolling back, a returning visitor can be sent straight to a `:443` that no longer listens, and will not retry `:80` until that cache entry has aged out. When you put TLS back, those clients recover on their own; if you need to check the current behaviour rather than a cached one, ask for a fresh copy with `curl -sI -H 'Cache-Control: no-cache' http://<domain>/`.

`NGINX_TLS` defaults to `auto`, and `auto` is deliberately biased towards the status quo: if the config already installed is serving `:443`, it keeps serving `:443`. Only `NGINX_TLS=off ./setup.sh nginx` removes TLS — a routine re-run never does, because a re-run cannot be sure what it is looking at. Concretely, TLS turns on when `LE_DOMAIN` is set and both `fullchain.pem` and `privkey.pem` are present and non-empty, **or** when the installed config already serves `:443`. A half-written pair from an interrupted run is not something to point a live server at, and that case does fall back to plain HTTP — with a loud warning. A certificate that is merely **expired** does not downgrade: it is still loaded, `./setup.sh tls` renews it (certbot treats a lapsed certificate as "until expiring"), and `./setup.sh nginx` — which runs no certbot at all — reports the expiry instead of removing the `:443` server. Trading a browser warning for cleartext is never the right side of that deal. A certificate that is not a certificate at all is reported as corrupt, with a different remedy, because `--keep-until-expiring` leaves such a file untouched and re-running the stage would loop forever.

`LE_DOMAIN` is read from the environment and written nowhere, so a re-run from a shell that does not export it recovers the domain from the certificate the **installed config** already names, and from nowhere else. That is deliberately the only source: `/etc/letsencrypt` is shared, so “whatever entry happens to be in `live/`” is not evidence that the certificate belongs to this site, and acting on it would serve a certificate for a domain the site does not answer for. The stage says so explicitly when it recovers — `serving: https via <domain> (domain recovered from the installed config; …)`. Export `LE_DOMAIN` for day-to-day work so this is never load-bearing.

**State of this repository: TLS is supported but NOT enabled.** No domain name is configured for this deployment, so `./setup.sh nginx` still writes the plain-HTTP config and the credentials-in-cleartext risk is unchanged until someone with the domain runs `./setup.sh tls`. Enabling it is an infrastructure decision, not a code one.

**Recommendation (not done here).** The header already carries `max-age=63072000` (two years, the commonly recommended minimum) and the `preload` token, but those only take effect over https. Submitting a domain to hstspreload.org is effectively irreversible — removal takes months — and `includeSubDomains` means every subdomain must serve HTTPS for the whole `max-age`. Do not submit until TLS is live on the apex *and* every subdomain, and treat it as its own change.

### Security

Hardening I baked into `setup.sh`:

- **Services bound to localhost** — Qdrant and Redis are published as `127.0.0.1:PORT:PORT` so they're only reachable from the host (nginx, the API), never the internet. If containers were previously created with public binds, `setup.sh backend` detects it and recreates them with the local bind (Qdrant's data volume is preserved).
- **API bound to localhost** — gunicorn binds `127.0.0.1:$API_PORT`, never the wildcard address. Search, chat, auth and analytics are reachable only through nginx on `:80`; nothing on `:8001` answers from off-host even if every firewall below is skipped. `setup.sh services` and `ecosystem.config.js` both use the loopback bind, and the nginx config writes `proxy_pass http://127.0.0.1:$API_PORT` to match. To apply the bind on an already-deployed host, run `./setup.sh services` — it deletes and re-registers the pm2 process, which is the only step that rewrites the stored argv. A bare `pm2 restart vccircle-backend` (including with `--update-env`, which refreshes environment variables but not the argument list) replays the argv pm2 stored at start time, so it keeps the old wildcard bind, and the enabled pm2 systemd unit resurrects that same argv from `~/.pm2/dump.pm2` after a reboot. Confirm with `ss -ltn | grep 8001` that the listening address is `127.0.0.1`.
- **Frontend bound to localhost** — `next start` binds `127.0.0.1:$NEXT_PORT`, never the wildcard address. Without an explicit `-H`, Next.js binds all interfaces (the Node default behind `server.listen(port, undefined)` is the wildcard `::`, dual-stack), which would publish the app shell, `/login`, `/signup` and `middleware.ts` on `:3000` to anything that can route to the host, bypassing nginx and with it TLS termination, the header and request-size limits, rate limiting and access logging. The flag is set in all three places the frontend can be started — `setup.sh services`, `ecosystem.config.js` and the `npm start` script — because each is effective on its own and none of them falls back to the others. `next dev` deliberately keeps the wildcard; it is a local development server, not the deployed one. As with the API, `pm2 restart vccircle-frontend` will not apply a changed `-H`: run `./setup.sh services` on an already-deployed host, and confirm with `ss -ltn | grep 3000` that the listening address is `127.0.0.1`.
- **Host firewall — MANDATORY** — a loopback bind is the primary control, not the only one: it does nothing for the ports that *are* meant to be public, and it fails open the moment someone re-binds a service to a wildcard address. UFW is a required deployment step, not a recommendation. Allow only SSH and HTTP, deny the rest:

```bash
sudo ufw default deny incoming
sudo ufw allow OpenSSH && sudo ufw allow 80/tcp
sudo ufw --force enable
```

  Also restrict the cloud security group (e.g. AWS) to ports 22/80.

- **nginx security headers** — `X-Content-Type-Options: nosniff`, `X-Frame-Options: DENY`, and `Referrer-Policy: strict-origin-when-cross-origin` on every location. CSP is set by the frontend (`middleware.ts`, per-request nonce), so I don't duplicate it at nginx. Transport security is opt-in: plain HTTP by default, `./setup.sh tls` to switch (see [TLS](#tls-https) above). HSTS is deliberately **not** set at nginx level — `next.config.ts` is the single source for that header, and two `max-age` values drift.
- **Pinned images** — Qdrant/Redis run from pinned, digest-resolvable tags (`QDRANT_IMAGE=qdrant/qdrant:v1.19.0@sha256:057e...d1fc`, `REDIS_IMAGE=redis:7-alpine`). If you override Qdrant, keep it >= the version that wrote any existing collection — older releases can't read newer storage formats.
- **API note** — CORS is restricted to the origins in `CORS_ORIGINS` (localhost dev origins by default; production is same-origin through nginx). **Auth** uses opaque bearer tokens with RBAC roles (see `app/auth.py`), and signup/login endpoints are rate-limited per client IP (Redis-backed). LLM spend is bounded by `LLM_DAILY_BUDGET_USD` (see `app/cost_budget.py`).
- **Health monitoring** — `deploy/healthcheck.sh` (I run it from cron every few minutes) probes `/ready/deep` for readiness and `/health` for liveness. Not ready while the process is alive → alert, but **no** restart, because a restart cannot bring Qdrant back or fix a placeholder `GEMINI_API_KEY` (the alert says so). It also says which probe was refused if the answer was 403/404/429/500 rather than 503, because none of those is a verdict about your dependencies. Not alive → `pm2 restart vccircle-backend`, then re-probe, then post an alert to `HEALTHCHECK_WEBHOOK_URL` if it has still not recovered. A fault alerts **once**: a repeat is logged as `still failing (live-not-ready:503)` and is otherwise silent until `ALERT_COOLDOWN_SECONDS` (default 3600) has passed — at `*/5` an unfixed fault would otherwise POST and mail the same news ~288 times a day, which is how the one alert that matters gets ignored. The dedup state lives beside the log in `logs/healthcheck.log.state` (override with `STATE_FILE`) and is written atomically; a corrupt, truncated or clock-skewed one is treated as "never alerted" rather than being allowed to silence anything. A healthy run clears it, so the next fault alerts immediately. The webhook is **best effort**: a failing POST is logged as a warning and never changes the run's exit code. Logs to `logs/healthcheck.log`. That recovery uses `pm2 restart`, which replays the stored argv — correct for reviving a sick process, but it cannot apply a changed bind or any other process option. Anything that changes the pm2 process definition (the API bind, the OOM auto-restart limits) needs `./setup.sh services`.

## Supported Settings

All optional (`backend/.env`), see `.env.example` for the full list:

| Variable | Default | Purpose |
| --- | --- | --- |
| `MYSQL_HOST/PORT/USER/PASSWORD/DATABASE/TABLE` | `localhost/3306/root//vccircle/articles` | Source DB (`vcc_frontend`, pk `feid`, `status=1`) |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `QDRANT_COLLECTION` | `vccircle_articles` | Collection name |
| `REDIS_URL` | `redis://localhost:6379/0` | Shared query cache (falls back to in-process cache if Redis is down) |
| `EMBED_MODEL` / `SPARSE_MODEL` | `BAAI/bge-base-en-v1.5` / `Qdrant/bm25` | Dense / sparse embedders (must match index time) |
| `EMBED_DENSE_CHAR_LIMIT` / `EMBED_CHAR_LIMIT` / `BODY_CHAR_LIMIT` | `1500` / `50000` / `50000` | Chars for the dense vector; sparse/lexical vector; body chars kept in the payload |
| `INDEXER_WORKERS` / `EMBED_BATCH_SIZE` | `2` / `256` | Encode/upsert pipeline depth; embedder batch size (each in-flight batch peaks ~1-2GB on CPU) |
| `EMBED_DEVICE` | `cpu` | `cuda` for a GPU |
| `RERANK_MODEL` / `RERANK_CANDIDATES` | `cross-encoder/ms-marco-MiniLM-L-6-v2` / `12` | Cross-encoder reranker; how many RRF candidates to re-score (12 keeps top-8 quality vs 16, faster on CPU). Clamped to `[5, 50]`: out-of-range values are clamped to the nearest bound and logged as a warning |
| `GEMINI_API_KEY` / `GEMINI_BASE_URL` / `LLM_MODEL` (`GEMINI_MODEL`) | — | Google Gemini (OpenAI-compatible endpoint) for chat |
| `LLM_PRICE_INPUT_PER_1M` / `LLM_PRICE_OUTPUT_PER_1M` / `INR_PER_USD` | `0.25` / `1.50` / `95.60` | USD per 1M input/output tokens (for cost display); USD→INR rate |
| `LLM_DAILY_BUDGET_USD` | `5.0` | Daily LLM spend cap; chat fails closed (refuses LLM calls) once today's cumulative cost reaches this value. Each turn reserves this cap atomically before a billed call, so concurrent turns contend for the same budget. Set `0` to deliberately disable the cap (see `app/cost_budget.py`) |
| `LLM_CALL_RESERVE_USD` | `0.05` | Budget a turn holds against the cap before each billed LLM call; the hold is reconciled to the real cost when the turn ends (see `app/cost_budget.py`) |
| `COST_RESERVATION_TTL_SECONDS` | `900` | Lifetime of an unsettled budget reservation. A hold that is never settled (a crashed turn) is swept after this long and **charged** to the day's spend at its reserved amount, not refunded — it leaves the holds table at the same time, so one crash can neither be free spend nor starve the cap until the day rolls over |
| `CHAT_MAX_HISTORY_CHARS` | `24000` | Total character budget for conversation history fed to the LLM; oldest turns are dropped once exceeded (see `CHAT_MAX_HISTORY_TURNS`) |
| `LLM_TIMEOUT_SECONDS` / `LLM_MAX_RETRIES` / `LLM_RETRY_BACKOFF` | `60` / `2` / `1.0` | LLM per-call timeout, retry count, exponential-backoff base |
| `TOP_K` / `ASK_MIN_SCORE` | `8` / `0.2` | Default result count; chat retrieval threshold |
| `CACHE_TTL_SECONDS` / `CACHE_MAX_SIZE` | `300` / `1000` | Query cache TTL; size of the in-process fallback cache |
| `CHAT_DB_PATH` / `CHAT_RETENTION_DAYS` / `CHAT_MAX_HISTORY_TURNS` | `data/chat.db` / `180` / `10` | SQLite chat store; idle-purge window; context turns kept per conversation |
| `RECENCY_STRENGTH` / `RECENCY_DECAY_DAYS` | `0.25` / `90` | Recency-tempered ranking blend |
| `RECENCY_BOOST_STRENGTH` / `RECENCY_BOOST_DECAY_DAYS` | `0.85` / `30.0` | Stronger recency blend applied when the query itself asks for fresh news ('latest'/'recent'/'fresh'); hard windows like 'this week' are filtered, not boosted |
| `WEAK_RESULT_SCORE` / `WEAK_RESULT_MIN_STRONG` | `0.3` / `3` | A hit counts as strong above `WEAK_RESULT_SCORE`; a result list with fewer than `WEAK_RESULT_MIN_STRONG` strong hits is reported as weakly answered (capped at the list length, floored at 1). Moves the `/search` weak note; chat's fallback only ever sees one source, where the count is always 1 |
| `DATE_FILLER_SCORE` | `0.2` | Relevance floor for date-only fallback fillers (temporal queries whose lexical signal is too weak to fill the window). Independent of `ASK_MIN_SCORE` — keep it at or above that gate, or the fillers are dropped before the model sees them |
| `ENABLE_QUERY_EXPANSION` / `ENABLE_ENTITY_BOOST` / `ENABLE_WEAK_FALLBACK` | `true` / `true` / `true` | Query-synonym expansion; entity-mention rerank boost; honest weak-result fallback (see `app/query_expand.py`, `app/rerank_boost.py`, `app/answer_fallback.py`) |
| `ANALYTICS_REDIS_DB` | `1` | Analytics aggregates live in Redis DB N (cache is DB 0); read endpoints are gated by the auth layer (admin role) |
| `CORS_ORIGINS` | `http://localhost:3000,http://localhost:8001` | Comma-separated allowed origins for CORS (production is same-origin through nginx) |
| `ALLOWED_HOSTS` | derived from `CORS_ORIGINS` + this box's hostname/addresses (including its default-route address) + `localhost,127.0.0.1,testserver` | Comma-separated hostnames the API answers to; a request with any other `Host` gets 400 (`TrustedHostMiddleware`), and the effective list is logged at startup. A wrong value here 400s the whole site; `*` is rejected. Set it when the API is reachable under a name none of the defaults cover, e.g. a registered public domain. The API also serves no `/docs`, `/redoc` or `/openapi.json` — see `docs/API.md` |

## Testing and CI

Backend tests (pytest, fully offline — mocked Qdrant/Redis/MySQL/LLM):

```bash
cd backend
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest
ruff check .
```

Coverage: query-intent/date parsing, facet filter construction, effective intent, ranking + recency, RAG DTO (no body leak), LLM config wiring, chat store (CRUD, ownership isolation, retention, token/cost stats), SSE streaming (small-talk short-circuit + full-turn deltas), index fingerprinting/delta/reconciliation, and cache TTL and degraded fallback.

`.github/workflows/ci.yml` runs three jobs on push/PR to `main`. Superseded
runs are cancelled (concurrency group keyed on workflow + ref) and every job has
an explicit timeout.

1. **backend** (`timeout-minutes: 30`) — Python 3.11, `ruff check .`, then `python -m pytest -rs`, which must report **zero skipped tests** (the job fails otherwise). lua5.1 is installed first: `tests/test_budget_lua.py` runs the real shipped `_BUDGET_LUA` under it and is the only thing that catches drift between that script and the Python spend-cap model, so a silent skip would leave the cap unverified. The job installs the *full* `requirements.txt` rather than a slimmed test set because `app/main.py` does `from fastembed import SparseTextEmbedding` at module scope and several test modules import `app.main`, so the suite cannot be collected without the runtime stack. (`app/encoders.py` and `app/reranker.py` import `fastembed`/`sentence_transformers` lazily inside their constructors, and their tests fake those modules in `sys.modules`.) There is deliberately **no `ruff format` gate** — `ruff format --check` reports files that would be reformatted, so enforcing it would mean reformatting the tree, not CI.
2. **frontend** (`timeout-minutes: 20`) — Node 22, `npm ci`, `npm run lint` (eslint), `npx tsc --noEmit`, `npm run build`, `npm test` (vitest).
3. **security** (`timeout-minutes: 20`) — `pip-audit` on both requirements files, `npm audit --audit-level=high`, and a gitleaks secret scan (binary pinned to 8.28.0, download SHA-256 verified) over the full history of the checked-out ref.

### Audit policy

**`pip-audit` fails the build on any advisory. There are no suppressions and no
`--ignore-vuln` flags.** The gate previously carried a 7-entry ignore list
covering starlette advisories (PYSEC-2026-1943/1941/161/2281/2280/249/248) that
were reachable only through `fastapi==0.115.0`, which capped starlette below 0.39
and made every fixed release (0.40.0 → 1.3.1) unreachable. Issue #331 raised
fastapi to 0.141.1 and pinned `starlette==1.7.0` explicitly. 0.141.1 is the
current release; the pin is safe because its starlette constraint is uncapped
(`starlette>=0.46.0`), so every fixed release is reachable. (0.133.0 was the
first release to drop the `<1.0.0` cap; 0.141.1 is simply the newest, and its
newer floor keeps it off starlette releases the advisories predate.)
Pinning the transitive directly is deliberate: starlette used to float, so the
fix would otherwise depend on whatever the resolver happened to pick rather
than on a reviewed value. `pip-audit -r backend/requirements.txt` is clean
with no ignore list, and a new advisory anywhere fails the build immediately.

The `transformers==5.10.1` pin (which fixes CVE-2026-4372 / CVE-2026-5241 /
CVE-2026-1839, and is why `optimum-onnx` is deliberately not installed) audits
**clean** — `pip-audit` reports no transformers findings. `requirements-dev.txt`
also audits clean.

**`npm audit --audit-level=high`** is the high boundary, which is the standard
CI posture: the current 2 moderate findings (GHSA-82fw-gwwq-j7x9,
`@vitest/mocker` path traversal; fix is vitest 5, a breaking change) sit below
it and do not fail the build. A new high/critical advisory does.

**gitleaks** runs the default rule set with **no allow-list and no
`.gitleaks.toml`** — nothing is excluded, so a real key in any tracked file
fails the job. The scan is pinned to `HEAD` rather than gitleaks' default
`--all`: a `fetch-depth: 0` checkout has every branch in the object store, and
`--all` reports secrets committed on unrelated branches, which would fail this
job for code a PR never touched. `HEAD` loses nothing — a `pull_request`
checkout is the merge commit, so its history contains both the branch and
everything it branched off from. This branch scans clean (0 leaks).

The scan reads committed history, not the working tree, so the gate was proved
by committing a throwaway RSA private key on a scratch clone: gitleaks reported
the leak and exited 1, and the identical command exits 0 without it. Note that
a bare `aws_access_key_id = AKIA...` line — placeholder *or* realistic — and a
synthetic `ghp_` token are both **not** flagged by the default 8.28.0 rules in
this shape, so neither is a valid probe for "the gate can fail". Only the
private-key block actually flipped it.
