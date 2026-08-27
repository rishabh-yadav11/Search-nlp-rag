"""Retrieval-quality eval for /search against a curated golden set.

Run from anywhere with the backend serving on localhost:8001:

    python3 scripts/search_quality_test.py            # top_k=8 (prod default)
    python3 scripts/search_quality_test.py 5          # custom top_k

Uses the PUBLIC /search endpoint (no LLM, no auth, deterministic). Each golden
query lists the feids of articles that are unambiguously relevant; the harness
scores retrieval against those expected ids with:

  * recall@k   - fraction of expected ids found in the top-k
  * nDCG@k     - binary-relevance discounted cumulative gain
  * MRR        - mean reciprocal rank of the first expected hit
  * hit rate   - fraction of queries with >= 1 expected id in the top-k

Run before/after a retrieval change and compare the aggregate line; this is the
deterministic signal for whether a change improved search quality.

The golden set is curated so every expected feid is clearly relevant to the
query. One deliberately hard case is included (id 17, demonetisation): it is
known-weak today and exists so the harness can measure future improvements.
"""
import argparse
import json
import math
import urllib.parse
import urllib.request

BASE = "http://localhost:8001"
DEFAULT_K = 8

GOLDEN = [
    {
        "id": 1,
        "category": "entity (Ola / SoftBank)",
        "q": "How did Ola Cabs raise funding from SoftBank?",
        "expected": [29052, 31716],
    },
    {
        "id": 2,
        "category": "year-in-review (Flashback 2024)",
        "q": "Top startup funding deals of 2024",
        "expected": [61174, 61215],
    },
    {
        "id": 3,
        "category": "topical (fintech VC this year)",
        "q": "What are the biggest venture capital deals in fintech this year?",
        "expected": [66909, 66502, 65644],
    },
    {
        "id": 4,
        "category": "event-year (covid lockdown)",
        "q": "What happened to edtech startups during the 2020 Covid-19 lockdown?",
        "expected": [42525],
    },
    {
        "id": 5,
        "category": "event (Techcircle DEMO India 2013)",
        "q": "Which companies presented at the Techcircle DEMO India 2013 event?",
        "expected": [11283],
    },
    {
        "id": 6,
        "category": "deep-body (Subbarao / 2008 RBI)",
        "q": "How did the 2008 global financial crisis change the way the Reserve Bank of India communicates?",
        "expected": [12751],
    },
    {
        "id": 7,
        "category": "date-filter year (Flashback 2025 deals)",
        "q": "List the top 5 biggest funding rounds in India in 2025",
        "expected": [65309],
    },
    {
        "id": 8,
        "category": "count (Flashback 2024 VC dealmaking)",
        "q": "How many venture capital deals happened in India in 2024?",
        "expected": [61239, 61225],
    },
    {
        "id": 9,
        "category": "topical (AI startup funding)",
        "q": "What is the latest news about AI startups raising funding in India?",
        "expected": [54141, 63497, 58558],
    },
    {
        "id": 10,
        "category": "fresh article (ASK exit Shriram)",
        "q": "Ask Property Fund exits Shriram Properties project",
        "expected": [67884],
    },
    {
        "id": 11,
        "category": "topical (venture debt)",
        "q": "Venture debt providers in India",
        "expected": [37122, 39839, 26029],
    },
    {
        "id": 12,
        "category": "fresh article (Bharat Value Fund)",
        "q": "Bharat Value Fund invests in ethnic sweets brand Dharwad Pedha",
        "expected": [67881],
    },
    {
        "id": 13,
        "category": "fresh article (Manipal / Kinder)",
        "q": "Manipal Health to buy Kinder Womens Hospital",
        "expected": [67883],
    },
    {
        "id": 14,
        "category": "entity (PhonePe valuation)",
        "q": "PhonePe funding round valuation",
        "expected": [54121, 53858, 45379],
    },
    {
        "id": 15,
        "category": "topical (recent IPOs)",
        "q": "What are the biggest IPOs in India recently?",
        "expected": [59429, 66219],
    },
    {
        "id": 16,
        "category": "count + sector (edtech 2023)",
        "q": "How many deals happened in the edtech sector in 2023?",
        "expected": [54567, 53773, 53743],
    },
    {
        "id": 17,
        "category": "KNOWN-WEAK (demonetisation x fintech)",
        "q": "How did demonetisation in 2016 impact fintech startups in India?",
        "expected": [32090, 27619],
    },
    # --- typo scenarios (correction should recover the same relevant ids) ---
    {
        "id": 18,
        "category": "typo: brand (Ola -> Olla)",
        "q": "How did Olla Cabs raise funding from SoftBank?",
        "expected": [29052, 31716],
    },
    {
        "id": 19,
        "category": "typo: brand (SoftBank -> Sotbank)",
        "q": "How did Ola Cabs raise funding from Sotbank?",
        "expected": [29052, 31716],
    },
    {
        "id": 20,
        "category": "typo: brand (PhonePe -> PonePe)",
        "q": "PonePe funding round valuation",
        "expected": [54121, 53858, 45379],
    },
    {
        "id": 21,
        "category": "typo: generic (providers -> provders)",
        "q": "Venture debt provders in India",
        "expected": [37122, 39839, 26029],
    },
    {
        "id": 22,
        "category": "typo: generic (Property -> Propery)",
        "q": "Ask Property Fund exits Shriram Propery project",
        "expected": [67884],
    },
]


def dcg(relevances: list[int]) -> float:
    return sum(r / math.log2(i + 2) for i, r in enumerate(relevances))


def ndcg(ids: list[int], expected: set[int]) -> float:
    rel = [1 if i in expected else 0 for i in ids]
    if not any(rel):
        return 0.0
    ideal = dcg([1] * min(len(ids), len(expected)))
    return dcg(rel) / ideal if ideal else 0.0


def run_one(g: dict, k: int) -> dict:
    url = f"{BASE}/search?top_k={k}&q=" + urllib.parse.quote(g["q"])
    try:
        with urllib.request.urlopen(url, timeout=90) as r:
            d = json.load(r)
    except Exception as e:  # noqa: BLE001 - never abort the whole run on one bad response
        print(f"[WARN] query {g['id']}: request/parse failed ({e}); skipping")
        return _invalid(g, f"request/parse error: {e}")

    if not isinstance(d, dict):
        print(f"[WARN] query {g['id']}: malformed response (not an object); skipping")
        return _invalid(g, "malformed response: not an object")
    results = d.get("results")
    if not isinstance(results, list):
        print(f"[WARN] query {g['id']}: missing/invalid 'results' field; skipping")
        return _invalid(g, "missing or non-list 'results'")

    ids = []
    for idx, res in enumerate(results):
        if not isinstance(res, dict):
            print(f"[WARN] query {g['id']}: result #{idx} is not an object; skipping entry")
            continue
        rid = res.get("id")
        if rid is None:
            print(f"[WARN] query {g['id']}: result #{idx} missing 'id'; skipping entry")
            continue
        if not isinstance(rid, int):
            try:
                rid = int(rid)
            except (TypeError, ValueError):
                print(f"[WARN] query {g['id']}: result #{idx} has non-int 'id' ({rid!r}); skipping entry")
                continue
        ids.append(rid)

    expected = set(g["expected"])
    found_ranks = [i + 1 for i, x in enumerate(ids) if x in expected]
    recall = len(found_ranks) / len(expected)
    mrr = 1.0 / found_ranks[0] if found_ranks else 0.0
    return {
        "id": g["id"],
        "category": g["category"],
        "recall": recall,
        "ndcg": ndcg(ids, expected),
        "mrr": mrr,
        "hit": 1 if found_ranks else 0,
        "found": found_ranks,
        "cached": d.get("cached"),
        "latency_ms": d.get("latency_ms"),
        "note": d.get("note"),
    }


def _invalid(g: dict, reason: str) -> dict:
    return {
        "id": g["id"],
        "category": g["category"],
        "recall": 0.0,
        "ndcg": 0.0,
        "mrr": 0.0,
        "hit": 0,
        "found": [],
        "cached": None,
        "latency_ms": None,
        "note": f"INVALID RESPONSE: {reason}",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Retrieval quality eval for /search")
    ap.add_argument("top_k", nargs="?", type=int, default=DEFAULT_K,
                    help=f"top-k window (default {DEFAULT_K})")
    args = ap.parse_args()
    k = max(1, min(args.top_k, 50))

    results = [run_one(g, k) for g in GOLDEN]
    n = len(results)
    avg = lambda f: sum(r[f] for r in results) / n

    print(f"{'id':>3} {'recall':>7} {'nDCG':>6} {'MRR':>6} {'hit':>4} {'cached':>7} {'ms':>6}  category")
    for r in results:
        print(f"{r['id']:>3} {r['recall']:>7.2f} {r['ndcg']:>6.3f} {r['mrr']:>6.3f} {r['hit']:>4} "
              f"{r['cached']!s:>7} {round(r['latency_ms'] or 0):>6}  {r['category']}")
        if r["note"]:
            print(f"      note: {r['note']}")
        if r["found"]:
            print(f"      expected found at rank: {r['found']}")
        else:
            print("      NO expected id in top-k")

    print(f"\n=== aggregate (top_k={k}, n={n}) ===")
    print(f"mean recall@{k}: {avg('recall'):.3f}")
    print(f"mean nDCG@{k}:   {avg('ndcg'):.3f}")
    print(f"mean MRR:        {avg('mrr'):.3f}")
    print(f"hit rate:        {avg('hit'):.3f}")
    print(f"cache hits:      {sum(1 for r in results if r['cached'])}/{n}")
    print(f"weak-notes:      {sum(1 for r in results if r['note'])}/{n}")
    failed = [r["id"] for r in results if r["hit"] == 0]
    print("queries with zero expected hits:", failed if failed else "none")


if __name__ == "__main__":
    main()
