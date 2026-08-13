# VCCircle Semantic Search

A hybrid retrieval + RAG search over the VCCircle article corpus. The full
dataset is indexed into Qdrant with dense (semantic) and sparse (BM25) vectors,
fused with Reciprocal Rank Fusion. A FastAPI service queries it and optionally
synthesizes cited answers with an LLM (Groq). A Next.js single-page UI ties it
together.

```
MySQL ──fetch_data──▶ articles.jsonl ──build_index──▶ Qdrant (dense + sparse)
   │                                                         ▲
   └─────────update_index (incremental new/edit/delete)──────┘
                                                              │
Browsers ◀── nginx ───▶ Next.js app ◀──(same origin)──▶ FastAPI ──▶ Qdrant
                           /search, /ask, /health                 Groq (for /ask)
```

## Repository layout

```
backend/
  app/                 FastAPI application (config + main)
  scripts/
    fetch_data.py      MySQL -> data/articles.jsonl (paginated, resumable)
    build_index.py     articles.jsonl -> Qdrant embeddings (checkpointed)
    update_index.py    incremental MySQL->Qdrant sync (new/edited/deleted)
    reset_index.py     drop the index + data files, start from zero
  requirements.txt
  .env.example         configuration template
frontend/              Next.js app (App Router + TypeScript)
  app/page.tsx         search/ask UI
  .env.local.example   API base URL template
```

## How it works

- **Hybrid retrieval** — every article is embedded twice at index time:
  `BAAI/bge-base-en-v1.5` (dense, cosine) and `Qdrant/bm25` sparse vectors
  with IDF. At query time both are searched in a single Qdrant prefetch and
  fused with RRF, using `"<title>. <summary>"` as the searchable text.
- **Reranking** — the RRF candidates are re-scored with a cross-encoder
  (`cross-encoder/ms-marco-MiniLM-L-6-v2`) so the final `top_k` reflects true
  relevance, and the score shown to the user is that reranked score (0-1).
- **Result ordering** — relevance first (rerank score desc), recency second
  (`published_date` desc as the tie-break, missing dates last).
- **Ask mode (RAG)** — retrieves articles above `ASK_MIN_SCORE`, packs them
  into a numbered context, and asks Groq (`llama-3.3-70b-versatile`) for an
  answer with inline `[n]` citations. If nothing clears the threshold it says
  so instead of guessing.
- **Caching** — Redis-backed JSON cache shared across workers (falls back to
  an in-process cache if Redis is down) for both `/search` and `/ask`, keyed
  by query + top_k.

## Prerequisites

- Python 3.10+
- Node.js 18.18+ (for the frontend)
- Docker (for Qdrant)

## 1. Backend setup

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # fill in MySQL creds + GROQ_API_KEY
```

Configuration is read from `.env` (see [Supported settings](#supported-settings)
below). Qdrant and Redis, local:

```bash
docker run -d --name qdrant -p 6333:6333 \
  -v $(pwd)/qdrant_data:/qdrant/storage --restart unless-stopped qdrant/qdrant
docker run -d --name redis -p 6379:6379 --restart unless-stopped redis:7
```

## 2. Build the index (run once)

```bash
cd backend
python scripts/fetch_data.py     # MySQL -> data/articles.jsonl
python scripts/build_index.py    # embed + upsert into Qdrant
```

Both are safe to interrupt and re-run: `fetch_data.py` resumes from the max id
already written, `build_index.py` resumes from a line-number checkpoint.
Downloads the embedding models on first run (dense + sparse, cached locally).

> Sparse vectors use Qdrant's `Modifier.IDF` schema. `build_index.py` detects and
> recreates a collection built with a mismatched config (you'd need to re-index).

## 3. Keep the index current (incremental)

`update_index.py` keeps Qdrant in sync with the database without touching the
running app. It fingerprints every published row and, on each run, embeds and
upserts new or edited articles and deletes removed ones. It never recreates the
collection and is safe to run while the API is live.

```bash
cd backend
python scripts/update_index.py --init   # seed once, AFTER a full build (no embedding)
python scripts/update_index.py          # scheduled runs
```

State (`last_id` + per-row fingerprints) lives in `backend/data/index_state.json`
(gitignored). When nothing changed a run is a near-free no-op — no model load —
and a `flock` prevents overlapping runs.

Every 15 minutes via cron, deprioritized with `nice`:

```
*/15 * * * * flock -n ~/search-nlp-rag/backend/data/update.lock \
  nice -n 15 ~/search-nlp-rag/backend/venv/bin/python \
  ~/search-nlp-rag/backend/scripts/update_index.py \
  >> ~/search-nlp-rag/logs/update_index.log 2>&1
```

## 4. Reset the index (start from zero)

`reset_index.py` drops the Qdrant collection and deletes the local data
artifacts (`articles.jsonl`, build checkpoint, incremental state) so the next
build starts from scratch. Embedding model caches, the venv and `.env` are kept.

```bash
cd backend
python scripts/reset_index.py               # interactive confirmation
python scripts/reset_index.py --yes         # skip confirmation
python scripts/reset_index.py --keep-data   # drop the collection only
```

It warns if the API is still listening (port 8001) — live queries 500 until the
index is rebuilt. After wiping, rebuild with
`fetch_data.py` → `build_index.py` → `update_index.py --init` → (re)start the API.

## 5. Run the API

```bash
cd backend
./venv/bin/gunicorn -k uvicorn.workers.UvicornWorker --workers 4 \
  --bind 0.0.0.0:8000 --timeout 120 app.main:app
```

| Endpoint | Description |
|---|---|
| `GET /search?q=...&top_k=8` | Hybrid semantic search (no LLM), cached |
| `GET /ask?q=...&top_k=8` | Search + cited LLM answer, cached |
| `GET /health` | Liveness |

```bash
curl "http://localhost:8000/search?q=fintech%20funding&top_k=3"
curl "http://localhost:8000/ask?q=who%20is%20investing%20in%20fintech&top_k=5"
```

## 6. Run the frontend

```bash
cd frontend
npm install
npm run dev                  # http://localhost:3000
# production:
npm run build && npm run start
```

The UI defaults to calling the API **same-origin** (`location.origin`), which
is right when nginx proxies the API paths on the app's own port (the deployment
pattern below). For local dev (`next dev` on :3000 with the API on :8000) set
`NEXT_PUBLIC_API_BASE=http://localhost:8000` — see `.env.local.example`.
`window.API_BASE` before page load overrides anything.

## Deployment (nginx)

Port map: Qdrant `6333` (internal), API `8001` (internal), Next.js `3000`
(internal, or a static build), nginx `80` (public). nginx serves the app and
proxies the API paths on the same origin so the UI works with zero CORS setup:

```nginx
server {
    listen 80;
    server_name _;

    location /search { proxy_pass http://127.0.0.1:8001; }
    location /ask    { proxy_pass http://127.0.0.1:8001; }
    location /health { proxy_pass http://127.0.0.1:8001; }
    location /       { proxy_pass http://127.0.0.1:3000; }
}
```

The API's CORS is wide open (`allow_origins=["*"]`) for POC convenience —
restrict it to the real frontend origin before exposing the API directly.

## Supported settings

All optional (`backend/.env`), see `.env.example` for the full list:

| Variable | Default | Purpose |
|---|---|---|
| `MYSQL_HOST/PORT/USER/PASSWORD/DATABASE/TABLE` | `localhost/3306/root//vccircle/articles` | Source DB (`vcc_frontend`, pk `feid`, `status=1`) |
| `QDRANT_URL` | `http://localhost:6333` | Qdrant endpoint |
| `QDRANT_COLLECTION` | `vccircle_articles` | Collection name |
| `REDIS_URL` | `redis://localhost:6379/0` | Shared query cache (falls back to in-process cache if Redis is down) |
| `EMBED_MODEL` / `SPARSE_MODEL` | `BAAI/bge-base-en-v1.5` / `Qdrant/bm25` | Dense / sparse embedders (must match index time) |
| `RERANK_MODEL` / `RERANK_CANDIDATES` | `cross-encoder/ms-marco-MiniLM-L-6-v2` / `16` | Cross-encoder reranker; how many RRF candidates to re-score |
| `EMBED_BATCH_SIZE` / `EMBED_DEVICE` | `256` / `cpu` | Indexing batch size; `cuda` for a GPU |
| `GROQ_API_KEY` / `GROQ_BASE_URL` / `LLM_MODEL` | — | Groq-compatible endpoint for `/ask` |
| `TOP_K` / `ASK_MIN_SCORE` | `8` / `0.2` | Default result count; `/ask` retrieval threshold |
| `CACHE_TTL_SECONDS` / `CACHE_MAX_SIZE` | `300` / `1000` | Query cache TTL; size of the in-process fallback cache |

## Production notes (POC shortcuts to fix before real prod)

1. **Embeddings and the reranker run on CPU.** Query-time embedding and
   cross-encoder re-scoring are offloaded off the event loop, so the API stays
   responsive, but latency is higher than a GPU-backed embedder. Set
   `EMBED_DEVICE=cuda` for lower latency.
2. **Cache is Redis-backed** and shared across gunicorn workers; if Redis is
   down it silently degrades to a per-worker in-process cache (which no longer
   benefits the other workers until Redis returns).
3. **No auth/rate limiting** on the API — add before exposing beyond localhost.
4. **`published_date` payload index** assumes MySQL returns a parseable
   date/datetime string; adjust the payload schema if your column type differs.
5. **Manual process start** in the reference deployment — gunicorn and
   `next start` won't survive a reboot; wrap them in systemd when staging.