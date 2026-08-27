"""Deep-body eval: verifies whole-body indexing and body-grounded chat.

Run from `backend/` with the venv python and the backend running:

    cd backend
    ./venv/bin/python scripts/deep_body_eval.py

Prerequisites (all on the deployment box): MySQL reachable (config from
`backend/.env`), the `vccircle_articles` Qdrant collection, and the backend
serving on localhost:8001 (needed for the search and chat levels). Uses a
deterministic seed so re-runs are comparable.

Three levels:
1. INDEX   - for deep-body articles (clean body > 6000 chars), pick terms that
             appear ONLY in the body region beyond 6000 chars (absent from
             title/summary/body-head) and check each is present in the stored
             sparse vector. This is the coverage only whole-body sparse
             indexing provides.
2. SEARCH  - for a sample of those articles, query the live /search endpoint
             with raw title + two validated deep terms; check top-8 rank.
3. CHAT    - ground a body-only query (Techcircle DEMO India 2013, article
             11283, whose summary is empty) through the live chat endpoint and
             verify the answer contains the deep-only facts. Costs a small LLM
             fee; also validates the chat body-rescue rerank.
"""
import asyncio
import json
import os
import random
import re
import sys
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.abspath("."))

from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient

from app.config import config
from app.index_text import clean

DEEP_CUTOFF = 6000
SEED = 42
# Machine-to-machine bypass for the internal eval scripts; must match the
# backend's AUTH_SERVICE_TOKEN in backend/.env.
SERVICE_TOKEN = os.getenv("AUTH_SERVICE_TOKEN", "")
DEEP_FACTS = ("makemytrip", "flight", "bizgain", "webmobi")


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def _extract_sparse_vector(vector, feid: int):
    """Safely pull the `sparse` named vector out of a Qdrant result.

    The stored collection uses named vectors (`{"dense": ..., "sparse": ...}`),
    so `rec.vector` is normally a dict. Depending on the qdrant_client version
    and query options it may instead be a `NamedVectors` object or `None`. This
    helper returns the sparse vector (anything with `.indices`) or `None`, and
    logs clearly rather than crashing on an unexpected shape.
    """
    if vector is None:
        print(f"  WARN  feid={feid}: record has no vector (None); skipping sparse check")
        return None
    if isinstance(vector, dict):
        sv = vector.get("sparse")
        if sv is None:
            print(f"  WARN  feid={feid}: sparse named vector missing from vector dict; skipping")
        return sv
    # NamedVectors-like object (pydantic model / object with attributes).
    try:
        return getattr(vector, "sparse", None)
    except Exception as exc:
        print(f"  WARN  feid={feid}: unexpected vector shape {type(vector)!r}: {exc}")
        return None


def _http_json(url: str, payload: dict | None = None, method: str | None = None,
               headers: dict | None = None, timeout: int = 120) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method or ("POST" if payload is not None else "GET"),
                                 headers=headers or {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def _coerce_score(value: object) -> float:
    if isinstance(value, bool):
        return 0
    if isinstance(value, (int, float)):
        return float(value)
    return 0


async def fetch_rows():
    import aiomysql
    pool = await aiomysql.create_pool(
        host=config.MYSQL_HOST, port=config.MYSQL_PORT,
        user=config.MYSQL_USER, password=config.MYSQL_PASSWORD,
        db=config.MYSQL_DATABASE, autocommit=True)
    async with pool.acquire() as conn, conn.cursor(aiomysql.DictCursor) as cur:
        # config.MYSQL_TABLE is a trusted deployment config value (not user
        # input); f-string interpolation here is safe only because it must
        # remain a fixed identifier. Do NOT interpolate any user-supplied value.
        await cur.execute(
            f"SELECT feid, title, summary, body FROM {config.MYSQL_TABLE} WHERE status=1")
        rows = await cur.fetchall()
    pool.close()
    await pool.wait_closed()
    return rows


def main() -> None:
    random.seed(SEED)

    rows = asyncio.run(fetch_rows())
    print(f"mysql rows (status=1): {len(rows)}")

    deep = []
    for r in rows:
        b = clean(r["body"])
        if len(b) > DEEP_CUTOFF:
            deep.append((r, b))
    print(f"deep-body articles (clean body > {DEEP_CUTOFF}): {len(deep)}")
    sample = random.sample(deep, min(28, len(deep)))

    sparse = SparseTextEmbedding(config.SPARSE_MODEL)

    # ---- collect candidate deep terms per article ----
    cand = {}
    for r, b in sample:
        title_toks = tokens(r["title"] or "")
        sum_toks = tokens(r["summary"] or "")
        head_toks = tokens(b[:DEEP_CUTOFF])
        deep_toks = tokens(b[DEEP_CUTOFF:]) - title_toks - sum_toks - head_toks
        cand[r["feid"]] = sorted((t for t in deep_toks if len(t) >= 5), key=lambda t: -len(t))

    all_terms = sorted({t for lst in cand.values() for t in lst})
    term_idx: dict[str, list[int]] = {}
    BATCH = 256
    for i in range(0, len(all_terms), BATCH):
        chunk = all_terms[i:i + BATCH]
        for t, emb in zip(chunk, sparse.embed(chunk)):
            term_idx[t] = [int(x) for x in emb.indices]

    # pick up to 3 validated terms (tokenizes to >=1 index) per article
    chosen: dict[int, list[str]] = {}
    for feid, terms in cand.items():
        picked = [t for t in terms if term_idx.get(t)]
        chosen[feid] = picked[:3]

    # ---- Part 1: index-level coverage ----
    qdrant = QdrantClient(url=config.QDRANT_URL, timeout=30)
    ids = [feid for feid in chosen if chosen[feid]]
    recs = qdrant.retrieve(collection_name=config.QDRANT_COLLECTION, ids=ids, with_vectors=True)
    by_id = {r.id: r for r in recs}

    checked = []  # (feid, term, present)
    for feid, terms in chosen.items():
        rec = by_id.get(feid)
        if not rec:
            continue
        sv = _extract_sparse_vector(rec.vector, feid)
        if sv is None:
            continue
        sv_idx = {int(i) for i in sv.indices}
        for t in terms:
            present = any(i in sv_idx for i in term_idx[t])
            checked.append((feid, t, present))
            if not present:
                print(f"  MISSING  feid={feid} term={t!r} indices={term_idx[t]}")

    ok = sum(1 for _, _, p in checked if p)
    print(f"\nINDEX-LEVEL: {ok}/{len(checked)} deep-body terms present in stored sparse vectors")

    # ---- Part 2: search with raw title + two validated deep terms ----
    def search(q: str):
        url = "http://localhost:8001/search?top_k=8&q=" + urllib.parse.quote(q)
        return _http_json(url, timeout=60)

    hits = []
    for r, b in sample[:7]:
        feid = r["feid"]
        terms = chosen.get(feid, [])[:2]
        if not terms:
            continue
        q = f"{(r['title'] or '').strip()} {' '.join(terms)}".strip()
        res = search(q)
        ids = [x["id"] for x in res.get("results", [])]
        rank = (ids.index(feid) + 1) if feid in ids else None
        hits.append((feid, rank, q))
        print(f"  q[feid={feid}] rank={rank} | {q[:95]}")

    in_top8 = sum(1 for _, rk, _ in hits if rk is not None and rk <= 8)
    ranked1 = sum(1 for _, rk, _ in hits if rk == 1)
    print(f"\nSEARCH: {in_top8}/{len(hits)} in top-8, {ranked1}/{len(hits)} ranked #1")
    qdrant.close()

    # ---- Part 3: chat grounding on a body-only article (11283) ----
    r11283 = next((r for r in rows if r["feid"] == 11283), None)
    if r11283 is None:
        print("\nCHAT: article 11283 not found; skipping")
        return
    b11283 = clean(r11283["body"])
    sum11283 = (r11283["summary"] or "").strip()
    print(f"\n11283 summary empty: {not sum11283} | body len: {len(b11283)}")
    for fact in DEEP_FACTS:
        in_head = fact in (sum11283 + b11283[:DEEP_CUTOFF]).lower()
        in_deep = fact in b11283[DEEP_CUTOFF:].lower()
        print(f"  fact {fact!r}: head/summary={in_head} deep-only={in_deep and not in_head}")

    headers = {"Content-Type": "application/json"}
    if SERVICE_TOKEN:
        headers["X-Service-Token"] = SERVICE_TOKEN
    base = "http://localhost:8001/api/chat"
    sess = _http_json(base + "/sessions", {"title": "deepeval"}, headers=headers)
    sid = sess.get("id") if isinstance(sess, dict) else None
    if not sid:
        print("\nCHAT: session id missing from response; skipping")
        return
    try:
        q = "What companies presented at the Techcircle DEMO India 2013 event?"
        d = _http_json(base + f"/sessions/{sid}/messages", {"content": q}, headers=headers)
        a = d.get("assistant") if isinstance(d, dict) else None
        if not a or not isinstance(a, dict):
            print("  WARN  chat response missing 'assistant'; skipping analysis")
            return
        raw_sources = a.get("sources") or []
        sources = [
            (s.get("id"), round(_coerce_score(s.get("score")), 3), (s.get("title") or "")[:60])
            for s in raw_sources if isinstance(s, dict)
        ]
        content = a.get("content", "")
        prompt_tokens = a.get("prompt_tokens")
        cost = a.get("cost")
        print(f"  chat answer len: {len(content)}"
              f" | tokens: {prompt_tokens if isinstance(prompt_tokens, (int, float)) else 'n/a'}"
              f" | cost INR: {round(cost, 3) if isinstance(cost, (int, float)) else 'n/a'}")
        for s in sources:
            print("  SOURCE", s[0], s[1], s[2])
        low = (a.get("content") or "").lower()
        print("  11283 in sources:", any(s[0] == 11283 for s in sources))
        for fact in DEEP_FACTS:
            print(f"  answer has {fact!r}:", fact in low)
    finally:
        try:  # best-effort cleanup; eval session deletion is optional
            _http_json(base + f"/sessions/{sid}", method="DELETE", headers=headers)
        except Exception:  # noqa: S110
            pass


if __name__ == "__main__":
    main()