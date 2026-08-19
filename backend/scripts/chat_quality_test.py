"""Chat quality test across diverse in-depth queries (golden set).

Run from anywhere with the backend serving on localhost:8001:

    python3 scripts/chat_quality_test.py

Prerequisites: backend running (nginx proxies /api/chat from the host). Each
query makes a real LLM call (billed against the daily budget), so this is not
free. Creates and deletes a fresh chat session per query.

Each query carries an expected outcome: a category, optional key terms that
must appear in the answer, an optional expected publication year for the
plain-year date-filter cases, and an optional `dataviz` flag asserting the
answer carries a valid ```dataviz JSON block (mirroring app.chat.parse_dataviz).
Prints a compact result table and flags queries with issues. Extend QUERIES as
new failure modes are found.
"""
import json
import os
import re
import urllib.request

from dotenv import load_dotenv

# Load backend/.env (run from backend/) so AUTH_SERVICE_TOKEN is picked up.
load_dotenv()

BASE = "http://localhost:8001/api/chat"
# Machine-to-machine bypass for the internal eval scripts; must match the
# backend's AUTH_SERVICE_TOKEN in backend/.env.
SERVICE_TOKEN = os.getenv("AUTH_SERVICE_TOKEN", "")

_DATAVIZ_RE = re.compile(r"```dataviz\s*\n(.*?)\n```", re.DOTALL)

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
    {
        "id": 12,
        "category": "dataviz: ranked list with values (explicit bar chart ask)",
        "q": "Show me a bar chart of the top 5 biggest funding rounds in India in 2025 with their values",
        "terms": [],
        "dataviz": True,
        "view": "bar",
    },
    {
        "id": 13,
        "category": "dataviz: count + yearly (line, explicit chart ask)",
        "q": "Show me a line chart of how many venture capital deals happened in India in 2024",
        "terms": ["2024"],
        "year": 2024,
        "dataviz": True,
        "view": "line",
    },
    {
        "id": 14,
        "category": "dataviz: sector breakdown (pie, explicit chart ask)",
        "q": "Show me a pie chart of the biggest deals by sector in India in 2025",
        "terms": [],
        "dataviz": True,
        "view": "pie",
    },
    {
        "id": 15,
        "category": "dataviz: count across years (explicit chart ask)",
        "q": "Show me a chart of how many funding rounds Ola has raised over the years and in which years",
        "terms": ["ola"],
        "dataviz": True,
    },
    {
        "id": 16,
        "category": "no dataviz: ranked question without a visual ask",
        "q": "top 10 ipo deals in 2025",
        "terms": [],
        "year": 2025,
        "no_dataviz": True,
    },
    {
        "id": 17,
        "category": "dataviz: explicit table request",
        "q": "Show me a table of the top 10 ipo deals in 2025",
        "terms": [],
        "year": 2025,
        "dataviz": True,
        "view": "table",
    },
]


def _to_float(v: object) -> float | None:
    """Coerce a dataviz cell to float (plain numbers or digit strings)."""
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        try:
            return float(v.replace(",", ""))
        except ValueError:
            return None
    return None


def _missing_cell(v: object) -> bool:
    if v is None:
        return True
    if isinstance(v, str):
        return v.strip().lower() in (
            "", "value not stated", "not stated", "n/a", "na", "n/d", "nil", "none",
            "unknown", "tbd", "to be decided", "to be determined", "—", "-", "--",
        )
    return False


def _valid_value_column(rows: list[list[object]], j: int) -> bool:
    """A value column holds a number in every non-missing cell and at least one
    number overall (empty cells are allowed for 'value not stated' rows)."""
    present = [r[j] for r in rows if not _missing_cell(r[j])]
    return bool(present) and all(_to_float(v) is not None for v in present)


def _parse_dataviz(text: str) -> dict | None:
    """Return the assistant's dataviz JSON block, or None when absent/invalid.

    Mirrors backend app.chat.parse_dataviz so the eval asserts the same
    contract the frontend renderer relies on."""
    m = _DATAVIZ_RE.search(text or "")
    if not m:
        return None
    try:
        d = json.loads(m.group(1))
    except (ValueError, TypeError):
        return None
    if not isinstance(d, dict):
        return None
    columns = d.get("columns")
    rows = d.get("rows")
    if not isinstance(columns, list) or not columns or not all(isinstance(c, str) for c in columns):
        return None
    if not isinstance(rows, list) or not rows or not all(isinstance(r, list) for r in rows):
        return None
    if any(len(r) != len(columns) for r in rows):
        return None
    vc = d.get("value_column")
    if not isinstance(vc, int) or isinstance(vc, bool) or not (0 <= vc < len(columns)):
        vc = None
        for j in range(len(columns)):
            if _valid_value_column(rows, j):
                vc = j
                break
    if vc is not None and not _valid_value_column(rows, vc):
        return None
    if vc is None and d.get("view") != "table":
        return None
    return {"columns": columns, "rows": rows, "value_column": vc, "format": d.get("format"),
            "kind": d.get("kind"), "view": d.get("view")}


def _req(url: str, payload: dict | None = None, method: str | None = None,
         timeout: int = 180) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"}
    if SERVICE_TOKEN:
        headers["X-Service-Token"] = SERVICE_TOKEN
    req = urllib.request.Request(
        url, data=data, method=method or ("POST" if payload is not None else "GET"),
        headers=headers)
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
    dv = _parse_dataviz(a["content"]) if (q.get("dataviz") or q.get("no_dataviz")) else None
    dv_ok = True
    if q.get("dataviz"):
        dv_ok = dv is not None
        if q.get("view") and dv is not None:
            dv_ok = dv.get("view") == q["view"]
    elif q.get("no_dataviz"):
        dv_ok = dv is None
    return {
        "id": q["id"],
        "category": q["category"],
        "llm": a["prompt_tokens"] > 0,
        "n_src": len(a["sources"]),
        "top": (a["sources"][0]["id"], round(a["sources"][0].get("score", 0), 3),
                a["sources"][0]["title"][:55]) if a["sources"] else None,
        "years": sorted(years),
        "year_ok": year_ok,
        "dv_ok": dv_ok,
        "dv": dv,
        "answer_len": len(a["content"]),
        "tokens": a["prompt_tokens"],
        "cost": round(a["cost"], 3),
        "latency": round(a["latency_ms"], 0),
        "note": a.get("note"),
        "missing_terms": missing,
    }


def main() -> None:
    results = [run(q) for q in QUERIES]
    print(f"{'id':>3} {'llm':>4} {'src':>4} {'yr_ok':>6} {'dv':>4} {'toks':>7} {'INR':>6} {'lat':>6}  category")
    for r in results:
        print(f"{r['id']:>3} {r['llm']!s:>4} {r['n_src']:>4} {r['year_ok']!s:>6} "
              f"{'ok' if r['dv_ok'] else 'MISS':>4} {r['tokens']:>7} {r['cost']:>6} "
              f"{r['latency']:>6}  {r['category']}")
        print(f"     top source: {r['top']}")
        print(f"     source years: {r['years']} | note: {r['note']}")
        if r["missing_terms"]:
            print(f"     MISSING TERMS: {r['missing_terms']}")
        if r["dv"]:
            dv = r["dv"]
            print(f"     dataviz: cols={dv['columns']} rows={len(dv['rows'])} "
                  f"vc={dv['value_column']} format={dv['format']!r}")
            for row in dv["rows"][:4]:
                print(f"       {row}")
        print()
    fails = [r["id"] for r in results if not r["llm"] or r["missing_terms"] or not r["year_ok"] or not r["dv_ok"]]
    print("QUERIES WITH ISSUES:", fails if fails else "none")


if __name__ == "__main__":
    main()