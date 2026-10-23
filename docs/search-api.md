# VCCircle Search API

Public HTTP reference for the read-only search surface: full-text search over
the VCCircle article corpus and the facet values used to filter it.

Verified against the live deployment on **2026-10-23**, deployed commit
`9405534` (identical to this checkout). The verification log is at the end.

## At a glance

| | |
|---|---|
| Base URL | `http://13.206.47.160` |
| Endpoints | `GET /search`, `GET /facets` |
| Authentication | None |
| Format | JSON (`application/json`) |
| Rate limit | 60 requests/minute per client IP, per endpoint |
| Transport | HTTP only; no HTTPS listener is currently available |
| OpenAPI / Swagger | Not published (`/docs`, `/openapi.json` return 404) |

There is no `/api` prefix. Call `http://13.206.47.160/search`, not
`http://13.206.47.160/api/search`.

The API is open and server-to-server callable. Browser-based clients on other
origins are blocked by CORS until the deployment adds their origin — see
[Browser use and CORS](#browser-use-and-cors).

## Quick start

```bash
curl -sS 'http://13.206.47.160/search?q=banks&top_k=1'
```

The response to that request (one result):

```json
{
  "query": "banks",
  "results": [
    {
      "id": 7522,
      "title": "Banking Is A Sector In Need Of Fresh Ideas",
      "url": "https://www.vccircle.com/banking-sector-need-fresh-ideas",
      "published_date": "2011-11-16T13:17:09",
      "category": "Others",
      "summary": "",
      "author_names": ["Luke Johnson"],
      "industry_names": ["General"],
      "dealtype_names": ["Others"],
      "tag_names": [],
      "content_type": "article",
      "score": 0.9884565609104584
    }
  ],
  "cached": false,
  "latency_ms": 113.31,
  "note": null
}
```

`latency_ms` is server-side time and excludes the network round trip. A repeat
of the 8-result `banks` request returned `"cached": true` in 64 ms end-to-end.

## `GET /search`

Hybrid retrieval: dense (semantic) and BM25 (sparse) search run in one Qdrant
query and are fused with Reciprocal Rank Fusion, then a cross-encoder reranks
the candidates. No LLM is involved and no article body text is returned.

### Parameters

| Name | Type | Required | Default | Rules |
|---|---|---|---|---|
| `q` | string | yes | — | 1–512 characters. Normalized before search; whitespace-only becomes `400 empty query`. |
| `top_k` | integer | no | `8` | 1–50. The server may return **more** than `top_k` for queries that ask for a count; otherwise it returns up to `top_k` matches. |
| `industry` | string | no | — | Comma-separated values, max 10 values, each max 200 characters. Exact, case-sensitive match. |
| `dealtype` | string | no | — | Same rules as `industry`. |
| `author` | string | no | — | Same rules. |
| `content_type` | string | no | — | Same rules. Observed values include `article` and `interview`. |
| `tag` | string | no | — | Same rules. |
| `from_date` | string | no | — | ISO 8601 date or date-time; filters `published_date` inclusively from that instant. |
| `to_date` | string | no | — | Same format; the time is widened to `23:59:59.999999` of that day, so a date-only value includes the whole day. |

Rules that apply to all five facet parameters:

- Commas separate values and mean **OR** within that field.
- Different facet fields are combined with **AND**.
- Values are matched against the indexed strings exactly, including case.
  `industry=finance` returns no matches; `industry=Finance` does.
- Unknown values are not an error: the response is `200` with an empty
  `results` array.
- More than 10 values, any value over 200 characters, or a raw facet string
  over 2010 characters returns `400`.
- Get valid `industry`, `dealtype`, and `tag` values from
  [`GET /facets`](#get-facets). There is no discovery endpoint for `author` or
  `content_type`; use values seen in results.

Date examples:

```bash
# Whole of 2024
curl -sS 'http://13.206.47.160/search?q=funding&from_date=2024-01-01&to_date=2024-12-31'

# Single instant, UTC because no offset is given
curl -sS 'http://13.206.47.160/search?q=funding&from_date=2024-06-01T09:30:00Z'
```

Accepted date forms include `YYYY-MM-DD`, `YYYY-MM-DDTHH:MM:SS`,
fractional seconds, `Z`, and explicit offsets such as `+05:30`. Date-only and
offset-less values are treated as UTC. An unparseable value returns `400` with
`{"detail":"invalid from_date"}` or `{"detail":"invalid to_date"}`. A range
where `from_date` is after `to_date` is not rejected; it returns no matches.

### Result count and pagination

There is **no pagination**: no `page`, `offset`, `cursor`, or total count.
`top_k` is the only size control, capped at 50.

`top_k` is a floor, not an exact count. The effective count is
`min(max(top_k, count implied by the query), 50)`. If the query text asks for a
number of items, the server raises the count up to 50. For example, `top_k=2`
with `q=best 10 fintech deals in 2024` returned 10 results, and `top_k=8` with
`q=top 20 fintech deals in 2024` returned 20. A normal keyword query returns
exactly `top_k` when enough matches exist.

For broader coverage, issue several narrower queries and merge them by `id`
rather than asking for more than 50 at once.

### Response

| Field | Type | Notes |
|---|---|---|
| `query` | string | The query after whitespace normalization. It is **not** the rewritten retrieval query, so a typo-corrected search still echoes the original spelling. |
| `results` | array | Ranked articles, possibly empty. |
| `cached` | boolean | `true` when the exact query, `top_k`, and filters were served from the server cache. |
| `latency_ms` | number | Server-side processing time in milliseconds. A cached result is typically 1–3 ms; an uncached search is typically 60–250 ms. |
| `note` | string or null | Human-readable advisory for weak or empty result sets. Safe to display; do not parse it as an error code. |

Each object in `results` has these fields. Only `published_date`, `category`,
and `content_type` are nullable; arrays are always present. The `note` field is
top-level, not part of a result.

| Field | Type | Notes |
|---|---|---|
| `id` | integer | Stable article identifier. |
| `title` | string | Article headline. |
| `url` | string | Absolute `https://www.vccircle.com/...` URL. |
| `published_date` | string or null | As indexed, normally `YYYY-MM-DDTHH:MM:SS` with **no timezone offset**. |
| `category` | string or null | One or more categories joined by commas, e.g. `"M&A,Private Equity,Venture Capital"`. |
| `summary` | string | Short excerpt; may be `""`. Never the full body. |
| `author_names` | string[] | May be empty. |
| `industry_names` | string[] | Indexed industries for the article. |
| `dealtype_names` | string[] | Indexed deal types. |
| `tag_names` | string[] | Indexed tags/entities. |
| `content_type` | string or null | E.g. `article`, `interview`. |
| `score` | number | Relevance signal. Higher is better, but not a probability and not comparable across queries. |

Article `body` is never returned. `summary` is a short excerpt; follow `url`
for the full article.

### How `score` behaves

- It is a blended relevance value, not a 0–1 probability. The deployed API
  returned scores above 1 (`1.246`), because click-driven boosting multiplies
  the reranker score.
- It is meaningful within one response, not across responses. Do not threshold
  it against a constant or compare scores from different queries.
- Results are returned in ranked order, but the order is **not** guaranteed to
  be strictly descending by `score`: diversity re-ranking can place a
  lower-scored article ahead of a higher-scored one.
- Date-only fallback hits, inserted to fill a temporal query, score exactly
  `0.2`.

### Caching

Responses are cached server-side for about 5 minutes (300 s), keyed by the
effective query, `top_k`, and filters. A repeated identical request within
that window returns `"cached": true` and a much lower `latency_ms`. Cached
responses are still counted against the rate limit.

The `cached` flag only reports the whole-response cache. A `false` value does
not mean every internal step ran; retrieval itself has a separate cache.

### Query understanding

The server interprets natural-language hints and can change retrieval without
changing the echoed `query`:

- Temporal phrases such as `2024`, `2023-2024`, `last year`, `this week`, or
  `Q3 2024` can apply an implicit date window.
- Sector and deal-type words can apply implicit `industry`, `dealtype`, or
  `content_type` filters.
- Explicit parameters always win over inferred ones.
- An inferred facet that produces no results is dropped and retrieval is
  retried without it. Explicit filters are never dropped.
- Superlative/count phrases such as `best 10` can raise the effective result
  count, as described above.

For deterministic behavior, pass explicit date and facet parameters and use a
plain keyword query.

## `GET /facets`

Returns the filter values currently present in the index.

```bash
curl -sS 'http://13.206.47.160/facets'
```

```json
{
  "industry": ["Brand Solutions", "Cleantech", "Commodities", "..."],
  "dealtype": ["Credit", "Investment Banking", "M&A", "Markets", "Others", "Private Equity", "Venture Capital"],
  "tags": ["IPO", "VCC Startups", "startups", "Sensex", "Nifty", "..."]
}
```

| Field | Contents |
|---|---|
| `industry` | Distinct industries, sorted ascending, at most 200. |
| `dealtype` | Distinct deal types, sorted ascending, at most 200. |
| `tags` | The 200 most frequent tags, ordered by frequency (not alphabetical). |
| `author`, `content_type` | Not returned. |

The endpoint has no parameters. It scans the whole collection on a cache miss,
so the first call after the 5-minute cache expires can take seconds (measured
2.58 s cold, 57 ms warm). Pass the returned strings verbatim as the matching
`/search` facet parameter; matching is case-sensitive.

## Rate limits

Both endpoints allow **60 requests per minute per client IP**. The window is a
fixed 60 seconds opened by the first request, not a sliding window.

- Exceeding it returns `429`:
  ```json
  {"detail": "Too many attempts. Please try again shortly."}
  ```
  with `Retry-After: 60`.
- `Retry-After` quotes the whole window, so treat it as an upper bound and add
  your own jitter.
- There are **no** `X-RateLimit-*` headers. Use the status code.
- The limit is per IP, not per caller or per key. Clients behind a shared NAT
  or corporate proxy share one bucket. There is currently no API key, account
  tier, or way for a caller to obtain a higher limit.
- If the limiter's Redis is unavailable, these endpoints fail closed with
  `503` and `Retry-After: 60` rather than serving unrated traffic.
- `/search` and `/facets` have separate buckets.

Live verification sent 65 requests to `/facets` in one window: the first 58
returned `200`, the remaining 7 returned `429` with `Retry-After: 60`. Earlier
calls in the same window accounted for the difference from exactly 60.

## Errors

| Status | Cause | Body |
|---|---|---|
| `400` | `q` is empty after whitespace normalization | `{"detail":"empty query"}` |
| `400` | Unparseable date | `{"detail":"invalid from_date"}` or `{"detail":"invalid to_date"}` |
| `400` | Facet limits exceeded | `{"detail":"too many industry values (maximum 10)"}` and similar |
| `400` | `Host` header not allowed | Plain text `Invalid host header` |
| `405` | Wrong method, e.g. `POST /search` | `{"detail":"Method Not Allowed"}` with `Allow: GET` |
| `422` | Parameter validation: missing `q`, `top_k` outside 1–50, `q` over 512 characters, non-integer `top_k`, etc. | `{"detail":[{"type":"...","loc":["query","top_k"],"msg":"..."}],"request_id":"..."}` |
| `429` | Rate limit exceeded | `{"detail":"Too many attempts. Please try again shortly."}` |
| `500` | Unhandled server error | `{"detail":"Internal server error","request_id":"..."}` |
| `503` | Rate limiter unavailable | `{"detail":"Rate limiter unavailable"}` |

Every response carries an `X-Request-ID` header. A caller-supplied value is
echoed back when it matches `^[A-Za-z0-9._-]{1,64}$`; otherwise the server
generates one. Include it when reporting a problem. `422` and `500` bodies
also contain `request_id`; `400`, `429`, and `503` bodies do not.

Unknown paths are not JSON: nginx falls through to the Next.js frontend, so a
typo such as `/serch` returns the frontend's HTML `404` page.

## Browser use and CORS

Cross-origin browser requests are **not** open to arbitrary sites. The server
allows only origins configured in `CORS_ORIGINS`; the deployed allow-list still
contains local development origins (for example, `http://localhost:3000`), not
the public address:

| Request origin | Observed response |
|---|---|
| `http://localhost:3000` | `Access-Control-Allow-Origin: http://localhost:3000` |
| `http://13.206.47.160` | No `Access-Control-Allow-Origin` (same-origin pages do not need it) |
| `https://example.com` | No `Access-Control-Allow-Origin`; preflight returned `400 Disallowed CORS origin` |

Server-side calls (`curl`, Python, Node, Go, backend services) are unaffected
by CORS and work from anywhere. A third-party browser app cannot call the API
until the operator adds its origin to `CORS_ORIGINS` and redeploys.

## Host header and transport

The API answers only when the `Host` header is in its allow-list. The public
address `13.206.47.160` works; an arbitrary value such as `Host: evil.example`
returns `400 Invalid host header`.

Only plain HTTP is available on the public address. `https://13.206.47.160`
timed out during verification, so never send credentials or sensitive data
(there are none required for these endpoints) and be aware that queries and
responses cross the network in cleartext.

## Integration examples

### Python

```python
import requests

response = requests.get(
    "http://13.206.47.160/search",
    params={"q": "fintech funding", "top_k": 5, "industry": "Finance,TMT"},
    timeout=15,
)
response.raise_for_status()
payload = response.json()

for article in payload["results"]:
    print(article["score"], article["published_date"], article["title"], article["url"])

if payload["note"]:
    print("advisory:", payload["note"])
```

### Node.js 18+

```js
const url = new URL("http://13.206.47.160/search");
url.searchParams.set("q", "fintech funding");
url.searchParams.set("top_k", "5");
url.searchParams.set("from_date", "2024-01-01");

const response = await fetch(url, { signal: AbortSignal.timeout(15_000) });
if (!response.ok) throw new Error(`search ${response.status}: ${await response.text()}`);
const { results, note } = await response.json();

for (const article of results) console.log(article.score, article.title, article.url);
```

### Handling the response

- Treat `results` as already ranked; do not re-sort by `score`.
- Use `id` to deduplicate across multiple queries.
- Display `note` when it is non-null; it explains an empty or weak result set.
- Back off on `429` using `Retry-After`, with jitter.
- Treat an empty `results` array as a valid answer, not an error.
- Ignore unknown JSON fields added in future versions.

## Limitations

- No authentication, API keys, per-key quotas, or usage dashboard.
- No pagination, total count, or cursor; 50 results per query maximum.
- No full article text; `summary` may be empty.
- Rate limit is a flat 60/min per IP for every caller.
- Cross-origin browser access is closed by default.
- HTTP only.
- `score` is not calibrated and can exceed 1.
- Natural-language date/facet inference means the same `q` can behave
  differently over time as the corpus and inference rules change.

## Verification log

Probes were run from an external host against `http://13.206.47.160` on
2026-10-23, with no authentication. The deployment reported commit `9405534`,
the same commit as this checkout, and `/ready` reported `qdrant`, `models`,
`redis`, and `llm` all `ok`.

| Probe | Status | Observed |
|---|---|---|
| `GET /health`, `GET /live` | `200` | `{"status":"ok"}` |
| `GET /ready` | `200` | `{"ready":true,...}` all four checks `ok` |
| `GET /search?q=banks` | `200` | 8 results, `cached:false`, 0.219 s end-to-end |
| Same request repeated | `200` | `cached:true`, 0.064 s end-to-end, 1.9 ms `latency_ms` |
| `top_k=3` / `top_k=0` / `top_k=51` | `200` / `422` / `422` | Bound messages quote 1 and 50 |
| `q` 513 characters | `422` | `String should have at most 512 characters` |
| `q=%20%20` | `400` | `{"detail":"empty query"}` |
| `from_date=notadate` | `400` | `{"detail":"invalid from_date"}` |
| `from_date=2024-01-01&to_date=2024-12-31` | `200` | Only 2024 articles returned |
| `industry=Finance` / `industry=finance` | `200` / `200` | 3 matches / 0 matches (case-sensitive) |
| `industry=Finance,TMT` | `200` | Multi-value OR works |
| 11 facet values | `400` | `too many industry values (maximum 10)` |
| Unknown facet value | `200` | Empty results with the generic no-match `note` |
| `content_type=interview` | `200` | Every result had `content_type:"interview"` |
| `q=best 10 fintech deals in 2024&top_k=2` | `200` | 10 results returned |
| `q=top 20 fintech deals in 2024&top_k=8` | `200` | 20 results returned |
| `q=banks` with `Host: evil.example` | `400` | `Invalid host header` (plain text) |
| `POST /search` | `405` | `Allow: GET` |
| 65 rapid `GET /facets` | `200` ×58, `429` ×7 | `Retry-After: 60` on the rejects |
| `GET /facets` cold / warm | `200` | 2.58 s cold, 57 ms warm; 39 industries, 7 deal types, 200 tags |
| `Origin: http://localhost:3000` | `200` | Origin reflected in `Access-Control-Allow-Origin` |
| `Origin: http://13.206.47.160` | `200` | No `Access-Control-Allow-Origin` (same origin) |
| `Origin: https://example.com` + preflight | `200` / `400` | No allow-origin header / `Disallowed CORS origin` |
| `GET /docs`, `/openapi.json`, `/serch` | `404` | Served by the Next.js frontend as HTML |
| `https://13.206.47.160/health` | no response | Connection timed out |

Not exercised live: the `503` limiter-outage path, the `500` handler, and
cache expiry (the 300 s TTL is taken from source; a call 25 minutes after a
cold call was cold again, which is consistent with it). Everything marked as a
source-derived rule above was checked against the deployed commit rather than
inferred from the probes alone.
