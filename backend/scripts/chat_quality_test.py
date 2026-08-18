"""Chat quality test across diverse in-depth queries (golden set).

Run from anywhere with the backend serving on localhost:8001:

    python3 scripts/chat_quality_test.py

Prerequisites: backend running (nginx proxies /api/chat from the host). Each
query makes a real LLM call (billed against the daily budget), so this is not
free. Creates and deletes a fresh chat session per query.

Each query carries an expected outcome: a category, optional key terms that
must appear in the answer, and an optional expected publication year for the
plain-year date-filter cases. Prints a compact result table and flags queries
with issues. Extend QUERIES as new failure modes are found.
"""
import json
import urllib.request

BASE = "http://localhost:8001/api/chat"
UID = "qualityeval-7"

QUERIES = [
    {
        "id": 1,
        "category": "deep-body retrospective (event-year + body-rescue)",
        "q": "What lessons did RBI governor Subbarao say central banks learned from the 2008 crisis?",
        "terms": ["2008", "communication", "moral", "liquidity", "reassuring"],
    },
    {
        "id": 3,
        "category": "event-year (demonetisation; must NOT date-filter to 2016)",
        "q": "How did demonetisation in 2016 impact fintech startups in India?",
        "terms": ["demonetisation", "fintech"],
    },
    {
        "id": 4,
        "category": "event-year (covid lockdown; must NOT date-filter to 2020)",
        "q": "What happened to edtech startups during the 2020 Covid-19 lockdown?",
        "terms": ["covid", "lockdown"],
    },
    {
        "id": 5,
        "category": "date-filter: plain year (2024)",
        "q": "Top startup funding deals of 2024",
        "terms": [],
        "year": 2024,
    },
    {
        "id": 6,
        "category": "date-filter: plain year (2025)",
        "q": "Best fintech investments in 2025",
        "terms": [],
        "year": 2025,
    },
    {
        "id": 7,
        "category": "weak-fallback niche (few strong matches)",
        "q": "Which game providers are listed as partners of Ninewins at non-GamStop betting sites?",
        "terms": ["hacksaw", "booongo", "microgaming"],
    },
    {
        "id": 8,
        "category": "normal topical (fintech funding)",
        "q": "What are the biggest venture capital deals in fintech this year?",
        "terms": ["fintech", "funding"],
    },
    {
        "id": 9,
        "category": "normal topical (AI startups)",
        "q": "What is the latest news about AI startups raising funding in India?",
        "terms": ["ai"],
    },
    {
        "id": 10,
        "category": "entity (Ola / SoftBank)",
        "q": "How did Ola Cabs raise funding from SoftBank?",
        "terms": ["ola", "softbank"],
    },
    {
        "id": 11,
        "category": "deep-body breadth (second retrospective)",
        "q": "How did the 2008 global financial crisis change the way the Reserve Bank of India communicates?",
        "terms": ["2008", "communicat"],
    },
]


def _req(url: str, payload: dict | None = None, method: str | None = None,
         timeout: int = 180) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, method=method or ("POST" if payload is not None else "GET"),
        headers={"Content-Type": "application/json", "X-User-Id": UID})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.load(resp)


def run(q: dict) -> dict:
    sid = _req(BASE + "/sessions", {"title": "quality-test"})["id"]
    try:
        d = _req(BASE + f"/sessions/{sid}/messages", {"content": q["q"]})
    finally:
        try:  # best-effort cleanup; eval session deletion is optional
            _req(BASE + f"/sessions/{sid}", method="DELETE")
        except Exception:  # noqa: S110
            pass
    a = d["assistant"]
    low = a["content"].lower()
    terms = q.get("terms", [])
    missing = [t for t in terms if t not in low]
    years = {s.get("published_date", "")[:4] for s in a["sources"] if s.get("published_date")}
    year_ok = True
    if q.get("year"):
        year_ok = years == {str(q["year"])}
    return {
        "id": q["id"],
        "category": q["category"],
        "llm": a["prompt_tokens"] > 0,
        "n_src": len(a["sources"]),
        "top": (a["sources"][0]["id"], round(a["sources"][0].get("score", 0), 3),
                a["sources"][0]["title"][:55]) if a["sources"] else None,
        "years": sorted(years),
        "year_ok": year_ok,
        "answer_len": len(a["content"]),
        "tokens": a["prompt_tokens"],
        "cost": round(a["cost"], 3),
        "latency": round(a["latency_ms"], 0),
        "note": a.get("note"),
        "missing_terms": missing,
    }


def main() -> None:
    results = [run(q) for q in QUERIES]
    print(f"{'id':>3} {'llm':>4} {'src':>4} {'yr_ok':>6} {'toks':>7} {'INR':>6} {'lat':>6}  category")
    for r in results:
        print(f"{r['id']:>3} {r['llm']!s:>4} {r['n_src']:>4} {r['year_ok']!s:>6} "
              f"{r['tokens']:>7} {r['cost']:>6} {r['latency']:>6}  {r['category']}")
        print(f"     top source: {r['top']}")
        print(f"     source years: {r['years']} | note: {r['note']}")
        if r["missing_terms"]:
            print(f"     MISSING TERMS: {r['missing_terms']}")
        print()
    fails = [r["id"] for r in results if not r["llm"] or r["missing_terms"] or not r["year_ok"]]
    print("QUERIES WITH ISSUES:", fails if fails else "none")


if __name__ == "__main__":
    main()