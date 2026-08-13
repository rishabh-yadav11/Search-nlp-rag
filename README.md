# VCCircle Semantic Search POC

Performance-oriented POC: full dataset (50k–200k articles), hybrid dense+sparse
search in Qdrant, async FastAPI, cached queries, optional LLM answer synthesis
with citations.

```
backend/   FastAPI app (app/), data scripts (scripts/), requirements, .env
frontend/  Next.js app (App Router, TypeScript) — app/ + package.json
```

## Setup

```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Qdrant, local
docker run -d -p 6333:6333 -v $(pwd)/qdrant_data:/qdrant/storage qdrant/qdrant

cp .env.example .env   # fill in MySQL creds + GROQ_API_KEY
```

## Build the index (run once, resumable)

```bash
cd backend
python scripts/fetch_data.py     # MySQL -> backend/data/articles.jsonl (paginated, resumable)
python scripts/build_index.py    # embed + upsert into Qdrant (checkpointed, resumable)
```

Both scripts are safe to interrupt and re-run — `fetch_data.py` resumes from
the max `id` already written, `build_index.py` resumes from a line-number
checkpoint file. `build_index.py` downloads the embedding models on first run:
`BAAI/bge-small-en-v1.5` (dense, via sentence-transformers) and `Qdrant/bm25`
(sparse, via fastembed).

> Note: sparse vectors use Qdrant's `Modifier.IDF` schema. If you re-run
> `build_index.py` against a collection created by an older version of this
> project, it detects the mismatch and recreates the collection automatically
> (you'll need to re-index).

## Run the API

```bash
cd backend
uvicorn app.main:app --reload --port 8000
```

- `GET /search?q=...&top_k=8` — hybrid semantic search, no LLM, cached
- `GET /ask?q=...&top_k=8` — search + cited LLM answer, cached
- `GET /health`

`/ask` only retrieves articles whose dense similarity is at least
`ASK_MIN_SCORE` (default `0.2`); if the filter removes everything it answers
"no sufficiently relevant articles" instead of guessing.

## Run the frontend

Next.js (App Router, TypeScript), on port 3000 by default:

```bash
cd frontend
npm install
npm run dev        # dev, http://localhost:3000
# or
npm run build && npm run start   # production
```

Open http://localhost:3000, type a query, toggle SEARCH vs ASK, hit Run (or
Enter). The UI calls the API at `http://localhost:8000` by default — override
with `NEXT_PUBLIC_API_BASE` (see `.env.local.example`) or `window.API_BASE`
before the page loads if the API is elsewhere. It uses `location.origin` when
the app is served same-origin with the API (e.g. behind nginx).

The API has CORS wide open (`allow_origins=["*"]`) so the browser can call it
from :3000 — fine for local POC use, tighten to your actual frontend origin
before deploying anywhere shared.

## Known POC shortcuts (fix before real prod)

1. **Embeddings run on CPU.** The dense (sentence-transformers) and sparse
   (fastembed BM25) models default to CPU; both are offloaded off the event
   loop in `hybrid_search`, so the API stays responsive, but latency is higher
   than a GPU-backed embedder would give you. Set `EMBED_DEVICE=cuda` (and add
   a GPU) for lower per-request latency.
2. **No re-ranker.** Top-k from fusion is returned as-is; add a cross-encoder
   pass if precision on the top few results matters.
3. **In-memory cache (`cachetools`)** — fine for a single process; swap for
   Redis if you run more than one API worker, since an in-process cache won't
   be shared across workers.
4. **No auth/rate limiting** on the API — add before exposing beyond localhost.
5. **`published_date` payload index** assumes MySQL returns a parseable
   date/datetime string; adjust the payload schema if your column type differs.
