# VCCircle New Search — API Reference

Base URL: `https://<host>/` (also reachable directly at `http://<host>:8001` on the
host itself; the public entrypoint is nginx on port 80).

All endpoints are `GET` and return JSON. There is currently **no authentication or
rate limiting** — the API is publicly reachable via nginx.

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

## `GET /ask`

Search + LLM-synthesized answer with inline `[n]` citations. Calls Groq
(`llama-3.3-70b-versatile`). **Costs LLM credits per uncached request.**

### Query parameters

Same as `/search`, except `top_k` range is `1..20`.

### Response

```json
{
  "query": "who is investing in Indian fintech?",
  "answer": "Several VCs are active in Indian fintech, including Accel [1] and Peak XV [2]...",
  "sources": [ { "id": ..., "title": ..., "url": ..., "published_date": ..., "category": ..., "summary": ..., "score": ... } ],
  "cached": false,
  "latency_ms": 8120.0,
  "note": null
}
```

Notes:
- Answers cite article numbers `[n]` that map to `sources`.
- If no result clears the relevance threshold, `/ask` returns an honest
  "couldn't find strong matches" message instead of fabricating.
- LLM failures return `503` after bounded retries with backoff.

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
    "groq": { "ok": true }
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

# Ask with citations (LLM)
curl "https://<host>/ask?q=who%20is%20investing%20in%20fintech&top_k=5"

# Facet values for filter autocomplete
curl "https://<host>/facets"
```

---

## Notes & limitations

- **Public exposure**: no auth / rate limiting yet. Restrict via nginx (basic auth /
  API key), an AWS security-group allowlist, or a rate limiter before broad use.
- **Data freshness**: the index is refreshed by an incremental sync every 15 minutes
  via cron (`update_index.py`).
- **Caching**: `/search` and `/ask` responses are cached (TTL 120s) keyed by effective
  query + filters. `cached: true` indicates a cache hit.
- Interactive OpenAPI docs are served by FastAPI at `/docs` on the internal API port.
