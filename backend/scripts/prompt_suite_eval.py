"""Run every prompt in prompt_suite_prompts.txt against live /search and
/api/chat, storing raw results for later quality judging.

Run from anywhere with the backend serving on localhost:8001:

    python3 scripts/prompt_suite_eval.py --suite search
    python3 scripts/prompt_suite_eval.py --suite chat

Prerequisites: backend running on localhost:8001 (the box also proxies /api/chat
through nginx). Chat turns make real LLM calls (billed against the daily
budget). Each chat prompt gets a fresh session that is deleted afterwards.

Results land in data/eval_results/prompt_suite_<timestamp>_<suite>.json — one
document per prompt with everything a judge needs: full result lists (search)
or full answers + sources + cost (chat), plus latency and retrieval notes. The
file is rewritten after every prompt so an interrupted run keeps partial data;
pass --start/--limit to resume.
"""

import argparse
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import UTC, datetime

from dotenv import load_dotenv

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(dotenv_path=os.path.join(_BACKEND_DIR, ".env"))

PROMPTS_FILE = os.path.join(_BACKEND_DIR, "scripts", "prompt_suite_prompts.txt")
RESULTS_DIR = os.path.join(_BACKEND_DIR, "data", "eval_results")
# Loopback defaults; overridable because the eval box may expose other ports.
SEARCH_BASE = os.getenv("EVAL_SEARCH_BASE", "http://localhost:8001/search")
CHAT_BASE = os.getenv("EVAL_CHAT_BASE", "http://localhost:8001/api/chat")
SERVICE_TOKEN = os.getenv("AUTH_SERVICE_TOKEN", "")


class ApiError(Exception):
    """HTTP failure with the response body preserved for diagnosis."""

    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:500]}")
        self.status = status
        self.body = body


# Deterministic placeholder substitutions so every templated prompt runs against
# a concrete entity VCCircle covers. Rotations spread different entities across
# repeated templates instead of hammering a single company.
_ROTATIONS = {
    "[Company Name]": ["Paytm", "Zomato", "Lenskart", "Zerodha", "Swiggy", "PhonePe", "Razorpay", "Byju's"],
    "[PE Firm Name]": ["Blackstone", "KKR"],
    "[VC Firm Name]": ["Peak XV Partners", "Accel", "Blume Ventures"],
    "[Person Name]": ["Kunal Shah", "Nithin Kamath"],
    "[Sector]": ["fintech", "edtech", "healthcare"],
    "[Topic]": ["quick commerce", "UPI payments"],
}
_FIXED = {
    "[PE Firm 1]": "Blackstone",
    "[PE Firm 2]": "KKR",
    "[Company 1]": "Paytm",
    "[Company 2]": "PhonePe",
    "[Investor 1]": "Accel",
    "[Investor 2]": "Peak XV Partners",
    "[year]": "2025",
    "[Company/Sector]": "the Indian quick-commerce sector",
}
_ROTATION_STATE: dict[str, int] = {}


def substitute_placeholders(prompt: str) -> str:
    for token, values in _ROTATIONS.items():
        if token in prompt:
            idx = _ROTATION_STATE.get(token, 0)
            _ROTATION_STATE[token] = idx + 1
            prompt = prompt.replace(token, values[idx % len(values)])
    for token, value in _FIXED.items():
        prompt = prompt.replace(token, value)
    return prompt


_SECTION_HEADERS = frozenset(
    {
        "Funding & Startup Deals",
        "Private Equity",
        "Venture Capital",
        "Mergers & Acquisitions",
        "Sector Analysis",
        "Industry-Specific Questions",
        "People & Leadership",
        "Trends & Market Intelligence",
        "Search, Analysis & Custom Queries",
    }
)


def load_prompts() -> list[dict]:
    """Read the prompt file: blank lines are skipped, known section titles set
    the category, and every other non-blank line is a prompt (most end with
    '?' or '.', a few are plain statements like 'Tell me everything about...')."""
    prompts = []
    section = "Company & Brand"
    with open(PROMPTS_FILE, encoding="utf-8") as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            if line in _SECTION_HEADERS:
                section = line
                continue
            prompts.append(
                {
                    "id": len(prompts) + 1,
                    "section": section,
                    "prompt": substitute_placeholders(line),
                    "raw_prompt": line,
                }
            )
    return prompts


def _req(url: str, payload: dict | None = None, method: str | None = None, timeout: int = 240) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"}
    if SERVICE_TOKEN:
        headers["X-Service-Token"] = SERVICE_TOKEN
    req = urllib.request.Request(
        url, data=data, method=method or ("POST" if payload is not None else "GET"), headers=headers
    )
    # Direct loopback opener: never route the service token through env proxies.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise ApiError(exc.code, body) from exc


def _delete(url: str) -> None:
    """DELETE that never fails on a 2xx response with an empty/non-JSON body."""
    try:
        _req(url, method="DELETE")
    except json.JSONDecodeError:
        pass


class ResultStore:
    """Rewrites the whole JSON document after each prompt so a long run can be
    interrupted at any point without losing earlier results."""

    def __init__(self, path: str, meta: dict):
        self.path = path
        self.doc = {"meta": meta, "results": []}
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self._flush()

    def add(self, entry: dict) -> None:
        self.doc["results"].append(entry)
        self._flush()

    def _flush(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.doc, fh, indent=1, ensure_ascii=False)
        os.replace(tmp, self.path)


def _results_filename(ts: str, suite: str) -> str:
    return os.path.join(RESULTS_DIR, f"prompt_suite_{ts}_{os.getpid()}_{suite}.json")


def run_search_suite(prompts: list[dict], start: int, limit: int | None) -> str:
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    meta = {
        "suite": "search",
        "endpoint": "/search",
        "started_utc": ts,
        "total_prompts": len(prompts),
        "resumed_from": start if start > 1 else None,
    }
    store = ResultStore(_results_filename(ts, "search"), meta)
    for p in prompts:
        if p["id"] < start:
            continue
        if limit is not None and p["id"] >= start + limit:
            break
        url = SEARCH_BASE + "?" + urllib.parse.urlencode({"q": p["prompt"], "top_k": 8})
        t0 = time.perf_counter()
        try:
            resp = _req(url)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            results = [
                {
                    "id": r.get("id"),
                    "title": r.get("title"),
                    "published_date": r.get("published_date"),
                    "industry": r.get("industry"),
                    "dealtype": r.get("dealtype"),
                    "score": r.get("score"),
                    "summary_excerpt": (r.get("summary") or "")[:400],
                }
                for r in resp.get("results", [])
            ]
            entry = {
                **p,
                "ok": True,
                "error": None,
                "result_count": len(results),
                "results": results,
                "cached": resp.get("cached"),
                "note": resp.get("note"),
                "latency_ms": round(elapsed_ms),
            }
        except Exception as exc:  # one failing prompt must not abort the run
            entry = {
                **p,
                "ok": False,
                "error": repr(exc),
                "result_count": 0,
                "results": [],
                "latency_ms": round((time.perf_counter() - t0) * 1000),
            }
        store.add(entry)
        status = "ok" if entry["ok"] else f"ERROR {entry['error']}"
        print(
            f"[{p['id']:>3}/{len(prompts)}] search {status} n={entry['result_count']} "
            f"lat={entry['latency_ms']}ms :: {p['prompt'][:80]}"
        )
    return store.path


def run_chat_suite(prompts: list[dict], start: int, limit: int | None) -> str:
    if not SERVICE_TOKEN:
        raise SystemExit(
            "AUTH_SERVICE_TOKEN missing from backend/.env — chat turns would all 401; aborting before any calls."
        )
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    meta = {
        "suite": "chat",
        "endpoint": "/api/chat",
        "started_utc": ts,
        "total_prompts": len(prompts),
        "resumed_from": start if start > 1 else None,
        "orphan_session_risk": 0,
    }
    store = ResultStore(_results_filename(ts, "chat"), meta)

    def note_orphan() -> None:
        meta["orphan_session_risk"] += 1
        print("WARN: failed to delete eval chat session (orphan risk)", flush=True)

    for p in prompts:
        if p["id"] < start:
            continue
        if limit is not None and p["id"] >= start + limit:
            break
        sid = None
        try:
            sid = _req(CHAT_BASE + "/sessions", {"title": "prompt-suite-eval"}).get("id")
        except Exception as exc:  # session create failed; nothing billed yet
            store.add(
                {
                    **p,
                    "ok": False,
                    "error": f"session create failed: {exc!r}",
                    "answer": None,
                    "sources": [],
                    "latency_ms": 0,
                }
            )
            print(f"[{p['id']:>3}/{len(prompts)}] chat ERROR session-create :: {p['prompt'][:80]}")
            continue
        if sid is None:
            note_orphan()
            store.add(
                {
                    **p,
                    "ok": False,
                    "error": "session create response missing id",
                    "answer": None,
                    "sources": [],
                    "latency_ms": 0,
                }
            )
            continue
        t0 = time.perf_counter()
        try:
            d = _req(CHAT_BASE + f"/sessions/{sid}/messages", {"content": p["prompt"]})
            elapsed_ms = (time.perf_counter() - t0) * 1000
            # Tolerate a degraded response shape: keep whatever arrived instead
            # of discarding a billed answer to a KeyError.
            a = (d.get("assistant") or {}) if isinstance(d, dict) else {}
            answer = a.get("content")
            sources = a.get("sources") or []
            entry = {
                **p,
                "ok": answer is not None,
                "error": None if answer is not None else "assistant message missing from response",
                "raw_response": None if answer is not None else d,
                "answer": answer,
                "sources": sources,
                "note": d.get("note") or a.get("note"),
                "n_sources": len(sources),
                "prompt_tokens": a.get("prompt_tokens"),
                "completion_tokens": a.get("completion_tokens"),
                "cost_inr": a.get("cost"),
                "latency_ms": round(elapsed_ms),
                "llm_used": (a.get("prompt_tokens") or 0) > 0,
            }
        except Exception as exc:  # one failing prompt must not abort the run
            entry = {
                **p,
                "ok": False,
                "error": repr(exc),
                "answer": None,
                "sources": [],
                "latency_ms": round((time.perf_counter() - t0) * 1000),
            }
        finally:
            try:
                _delete(CHAT_BASE + f"/sessions/{sid}")
            except Exception:
                note_orphan()
        store.add(entry)
        status = "ok" if entry["ok"] else f"ERROR {entry['error']}"
        print(
            f"[{p['id']:>3}/{len(prompts)}] chat {status} src={entry.get('n_sources')} "
            f"lat={entry['latency_ms']}ms :: {p['prompt'][:80]}"
        )
    return store.path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    # Chat intentionally requires an explicit flag: every turn is a billed LLM
    # call, so a bare invocation must never burn the daily budget.
    ap.add_argument("--suite", choices=["search", "chat"], default="search")
    ap.add_argument("--start", type=int, default=1, help="first prompt id to run")
    ap.add_argument("--limit", type=int, default=None, help="how many prompts to run")
    args = ap.parse_args()

    prompts = load_prompts()
    print(f"loaded {len(prompts)} prompts; suite={args.suite} start={args.start} limit={args.limit}")
    runner = run_search_suite if args.suite == "search" else run_chat_suite
    print("results:", runner(prompts, args.start, args.limit))


if __name__ == "__main__":
    main()
