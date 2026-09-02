"""Unified chat eval runner over prompts_with_variations_flat.json.

For every prompt: create a chat session, POST one message, capture the answer,
sources, token usage and cost, fetch the cited articles' full bodies from
Qdrant, then delete the session. The JSON document is rewritten after every
prompt so an interrupted run keeps partial data; pass --start/--limit to resume.
A companion markdown report aggregates success/latency/cost, cross-variation
consistency (prompts are grouped by normalized text similarity) and anomalies.

Prerequisites: backend on localhost:8001 with AUTH_SERVICE_TOKEN in
backend/.env, plus Qdrant reachable at QDRANT_URL. Chat turns make real (billed)
LLM calls against the daily budget — use --dry-run to validate loading only.

    ./venv/bin/python scripts/eval_runner.py --dry-run
    ./venv/bin/python scripts/eval_runner.py --start 1 --limit 5
    ./venv/bin/python scripts/eval_runner.py --input /path/to/prompts.json

Results land in eval_results/<timestamp>.json and <timestamp>_report.md,
relative to the working directory (override with EVAL_RESULTS_DIR).
"""

import argparse
import json
import os
import re
import statistics
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from dotenv import load_dotenv

if TYPE_CHECKING:
    from qdrant_client import QdrantClient

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(dotenv_path=os.path.join(_BACKEND_DIR, ".env"))

CHAT_BASE = os.getenv("EVAL_CHAT_BASE", "http://localhost:8001/api/chat")
SERVICE_TOKEN = os.getenv("AUTH_SERVICE_TOKEN", "")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "vccircle_articles")
RESULTS_DIR = os.getenv("EVAL_RESULTS_DIR", "eval_results")
DEFAULT_INPUT = "prompts_with_variations_flat.json"
REQUEST_TIMEOUT = 240
HIGH_LATENCY_MS = 60_000
COST_OUTLIER_FACTOR = 3.0
SIMILARITY_THRESHOLD = 0.5
VARIATION_RUN = 4

_CONTRACTIONS = {
    "what's": "what is",
    "who's": "who is",
    "who're": "who are",
    "i'd": "i would",
    "let's": "let us",
}
_LEADING_FILLERS = (
    "i would like to know",
    "kindly",
    "please",
    "can you",
    "could you",
    "would you",
    "tell me",
    "show me",
    "give me",
    "let me know",
    "hey",
    "so",
    "what is",
    "what are",
    "who are",
    "state",
    "provide",
    "identify",
    "describe",
    "list",
    "name the",
    "names of the",
    "names of",
    "name",
    "indicate",
    "comprehensive",
    "the",
    "a",
    "an",
)
_STOPWORDS = frozenset(
    (
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "am", "of", "in", "on", "at", "by",
        "for", "with", "about", "over", "under", "to", "from", "into", "and", "or", "but", "has", "have", "had",
        "having", "how", "what", "who", "whom", "when", "which", "why", "where", "does", "do", "did", "doing",
        "can", "could", "would", "should", "may", "might", "will", "shall", "must", "i", "me", "my", "we", "our",
        "you", "your", "it", "its", "this", "that", "these", "those", "there", "here", "they", "them", "their",
        "like", "know", "tell", "show", "give", "let", "us", "state", "provide", "identify", "describe", "list",
        "name", "names", "comprehensive", "please", "hey", "kindly", "want", "need", "most", "very",
    )
)


class ApiError(Exception):
    """HTTP failure with the response body preserved for diagnosis."""

    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:500]}")
        self.status = status
        self.body = body


def _req(url: str, payload: dict | None = None, method: str | None = None, timeout: int = REQUEST_TIMEOUT) -> dict:
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"}
    if SERVICE_TOKEN:
        headers["X-Service-Token"] = SERVICE_TOKEN
    req = urllib.request.Request(
        url, data=data, method=method or ("POST" if payload is not None else "GET"), headers=headers
    )
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


def load_prompts(path: str) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        prompts = json.load(fh)
    if not isinstance(prompts, list) or not all(isinstance(p, str) and p.strip() for p in prompts):
        raise ValueError(f"{path}: expected a JSON list of non-empty prompt strings")
    return prompts


def _strip_leading_fillers(text: str) -> str:
    while True:
        stripped = next(
            (text[len(filler) :].strip() for filler in _LEADING_FILLERS if text == filler or text.startswith(filler + " ")),
            None,
        )
        if stripped is None:
            return text
        text = stripped


def normalize_prompt(prompt: str) -> str:
    text = prompt.lower()
    for contracted, expanded in _CONTRACTIONS.items():
        text = text.replace(contracted, expanded)
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    return _strip_leading_fillers(text)


def _content_tokens(normalized: str) -> frozenset[str]:
    return frozenset(t for t in normalized.split() if t not in _STOPWORDS)


def _jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    return len(a & b) / len(a | b)


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, i: int) -> int:
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a: int, b: int) -> None:
        self.parent[self.find(a)] = self.find(b)


def group_prompts(prompts: list[str]) -> dict[str, list[int]]:
    """Map group label -> 1-based prompt indices. The flat input lists the
    variations of each base prompt in consecutive runs of VARIATION_RUN, and a
    run's members restate identical content behind interchangeable scaffolding
    ("Tell me X" / "Hey, tell me X" / "List the most recent X"). So merging is
    confined to a run — pairwise, on filler-stripped normalized prompts whose
    content-word Jaccard similarity reaches SIMILARITY_THRESHOLD — which keeps
    neighboring runs about related topics (funding deals vs M&A deals) apart."""
    keys = [normalize_prompt(p) for p in prompts]
    tokens = [_content_tokens(k) for k in keys]
    uf = _UnionFind(len(prompts))
    for run_start in range(0, len(prompts), VARIATION_RUN):
        run = range(run_start, min(run_start + VARIATION_RUN, len(prompts)))
        for i in run:
            for j in run:
                if j > i and tokens[i] and tokens[j] and _jaccard(tokens[i], tokens[j]) >= SIMILARITY_THRESHOLD:
                    uf.union(i, j)
    clusters: dict[int, list[int]] = {}
    for i in range(len(prompts)):
        clusters.setdefault(uf.find(i), []).append(i + 1)
    grouped: dict[str, list[int]] = {}
    for members in clusters.values():
        label = min((keys[m - 1] for m in members), key=lambda k: (len(k), k))
        grouped[label] = sorted(members)
    return dict(sorted(grouped.items(), key=lambda kv: kv[1][0]))


class ResultStore:
    """Rewrites the whole JSON document after each prompt so a long run can be
    interrupted at any point without losing earlier results."""

    def __init__(self, path: str, meta: dict):
        self.path = path
        self.doc = {"meta": meta, "results": []}
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self._flush()

    def add(self, entry: dict) -> None:
        self.doc["results"].append(entry)
        self._flush()

    def _flush(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.doc, fh, indent=1, ensure_ascii=False)
        os.replace(tmp, self.path)


def _failure_entry(index: int, group: str | None, prompt: str, error: str, session_id: str | None = None) -> dict:
    return {
        "index": index,
        "group": group,
        "prompt": prompt,
        "session_id": session_id,
        "ok": False,
        "error": error,
        "answer": None,
        "sources": [],
        "bodies": {},
        "body_error": None,
        "n_sources": 0,
        "prompt_tokens": None,
        "completion_tokens": None,
        "cost_inr": None,
        "latency_ms": 0,
        "note": None,
    }


def _citation_ids(sources: list) -> list[int]:
    return sorted({s["id"] for s in sources if isinstance(s, dict) and isinstance(s.get("id"), int)})


def _fetch_bodies(client: "QdrantClient", source_ids: list[int]) -> tuple[dict[str, str], str | None]:
    try:
        points = client.retrieve(collection_name=QDRANT_COLLECTION, ids=source_ids, with_payload=True)
        return {str(p.id): (p.payload or {}).get("body", "") for p in points}, None
    except Exception as exc:
        return {}, repr(exc)


def run_prompt(client: "QdrantClient", index: int, group: str | None, prompt: str, sid: str) -> dict:
    t0 = time.perf_counter()
    try:
        d = _req(CHAT_BASE + f"/sessions/{sid}/messages", {"content": prompt})
    except Exception as exc:
        return _failure_entry(index, group, prompt, repr(exc), session_id=sid)
    latency_ms = round((time.perf_counter() - t0) * 1000)
    # Tolerate a degraded response shape: keep whatever arrived instead of
    # discarding a billed answer to a KeyError.
    a = (d.get("assistant") or {}) if isinstance(d, dict) else {}
    answer = a.get("content")
    sources = a.get("sources") or []
    source_ids = _citation_ids(sources)
    bodies, body_error = _fetch_bodies(client, source_ids) if source_ids else ({}, None)
    return {
        "index": index,
        "group": group,
        "prompt": prompt,
        "session_id": sid,
        "ok": answer is not None,
        "error": None if answer is not None else "assistant message missing from response",
        "raw_response": None if answer is not None else d,
        "answer": answer,
        "sources": sources,
        "bodies": bodies,
        "body_error": body_error,
        "n_sources": len(sources),
        "prompt_tokens": a.get("prompt_tokens"),
        "completion_tokens": a.get("completion_tokens"),
        "cost_inr": a.get("cost"),
        "latency_ms": latency_ms,
        "note": d.get("note") or a.get("note"),
    }


def run_eval(
    prompts: list[str],
    groups: dict[str, list[int]],
    input_path: str,
    start: int,
    limit: int | None,
    delay: float,
) -> ResultStore:
    if not SERVICE_TOKEN:
        raise SystemExit(
            "AUTH_SERVICE_TOKEN missing from backend/.env — chat turns would all 401; aborting before any calls."
        )
    from qdrant_client import QdrantClient

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    group_of = {index: label for label, members in groups.items() for index in members}
    meta = {
        "input": input_path,
        "endpoint": CHAT_BASE,
        "started_utc": ts,
        "total_prompts": len(prompts),
        "resumed_from": start if start > 1 else None,
        "delay_seconds": delay,
        "n_groups": len(groups),
        "orphan_session_risk": 0,
    }
    store = ResultStore(os.path.join(RESULTS_DIR, f"{ts}.json"), meta)
    qdrant = QdrantClient(url=QDRANT_URL, timeout=30)
    index = start
    interrupted = False
    try:
        for index, prompt in enumerate(prompts, start=1):
            if index < start:
                continue
            if limit is not None and index >= start + limit:
                break
            group = group_of.get(index)
            sid = None
            try:
                sid = _req(CHAT_BASE + "/sessions", {"title": "eval-runner"}).get("id")
            except Exception as exc:
                entry = _failure_entry(index, group, prompt, f"session create failed: {exc!r}")
            else:
                if isinstance(sid, str):
                    try:
                        entry = run_prompt(qdrant, index, group, prompt, sid)
                    except Exception as exc:
                        entry = _failure_entry(index, group, prompt, f"message phase failed: {exc!r}", session_id=sid)
                else:
                    entry = _failure_entry(index, group, prompt, "session create response missing id")
            finally:
                if isinstance(sid, str):
                    try:
                        _delete(CHAT_BASE + f"/sessions/{sid}")
                    except Exception:
                        meta["orphan_session_risk"] += 1
                        print("WARN: failed to delete eval chat session (orphan risk)", flush=True)
            store.add(entry)
            status = "ok" if entry["ok"] else f"ERROR {entry['error']}"
            print(
                f"[{index:>3}/{len(prompts)}] {status} src={entry.get('n_sources')} "
                f"lat={entry.get('latency_ms')}ms :: {prompt[:80]}",
                flush=True,
            )
            time.sleep(delay)
    except KeyboardInterrupt:
        interrupted = True
    finally:
        qdrant.close()
    if interrupted:
        print(f"interrupted at prompt {index}; resume with --start {index}", flush=True)
    return store


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(0.95 * (len(ordered) - 1)))]


def _pairwise_source_jaccard(results: list[dict]) -> float | None:
    sets = [frozenset(_citation_ids(r.get("sources") or [])) for r in results]
    pairs = [(a, b) for i, a in enumerate(sets) for b in sets[i + 1 :] if a or b]
    if not pairs:
        return None
    return statistics.mean(len(a & b) / len(a | b) for a, b in pairs)


def build_report(store: ResultStore) -> str:
    meta = store.doc["meta"]
    results = store.doc["results"]
    ok = [r for r in results if r.get("ok")]
    latencies = [r["latency_ms"] for r in ok if isinstance(r.get("latency_ms"), (int, float))]
    costs = [r["cost_inr"] for r in ok if isinstance(r.get("cost_inr"), (int, float)) and r["cost_inr"]]
    median_cost = statistics.median(costs) if costs else None

    by_group: dict[str, list[dict]] = {}
    for r in ok:
        by_group.setdefault(r.get("group") or "?", []).append(r)
    consistency_rows = []
    within_jaccards = []
    for label, members in sorted(by_group.items()):
        jac = _pairwise_source_jaccard(members)
        if jac is not None:
            within_jaccards.append(jac)
        lengths = [len(r.get("answer") or "") for r in members]
        stdev = round(statistics.stdev(lengths), 1) if len(lengths) > 1 else None
        consistency_rows.append((label, len(members), jac, round(statistics.mean(lengths)), stdev))

    anomalies = []
    for r in results:
        label = f"#{r['index']} ({r.get('group')})"
        if not r.get("ok"):
            anomalies.append(f"{label}: failed — {r.get('error')}")
            continue
        if not (r.get("answer") or "").strip():
            anomalies.append(f"{label}: empty answer")
        if r.get("n_sources") == 0:
            anomalies.append(f"{label}: zero sources")
        if r.get("body_error"):
            anomalies.append(f"{label}: body fetch failed — {r['body_error']}")
        if isinstance(r.get("latency_ms"), (int, float)) and r["latency_ms"] > HIGH_LATENCY_MS:
            anomalies.append(f"{label}: high latency {r['latency_ms']}ms")
        if (
            median_cost
            and isinstance(r.get("cost_inr"), (int, float))
            and r["cost_inr"] > COST_OUTLIER_FACTOR * median_cost
        ):
            anomalies.append(f"{label}: cost outlier {r['cost_inr']} INR (median {median_cost:.2f})")

    def fmt(value) -> str:
        return "n/a" if value is None else f"{value:.3f}" if isinstance(value, float) else str(value)

    lines = [
        f"# Eval report — {meta['started_utc']}",
        "",
        f"- Input: `{meta.get('input')}` | endpoint `{meta.get('endpoint')}`",
        f"- Prompts run: {len(results)} of {meta['total_prompts']}"
        + (f" (resumed from {meta['resumed_from']})" if meta.get("resumed_from") else ""),
        f"- Variation groups: {meta.get('n_groups')} | orphan sessions: {meta.get('orphan_session_risk', 0)}",
        "",
        "## Aggregate stats",
        "",
        f"- Success: {len(ok)}/{len(results)} ({100 * len(ok) / len(results):.1f}%)" if results else "- No results",
        (
            f"- Latency ms — p50: {fmt(statistics.median(latencies) if latencies else None)}, "
            f"p95: {fmt(_p95(latencies))}, max: {fmt(max(latencies) if latencies else None)}"
        ),
        (
            f"- Total cost: {sum(costs):.2f} INR | prompt tokens: {sum(r.get('prompt_tokens') or 0 for r in ok)}, "
            f"completion tokens: {sum(r.get('completion_tokens') or 0 for r in ok)}"
        ),
        "",
        "## Cross-variation consistency",
        "",
        (
            f"Groups with comparable answers: {len(consistency_rows)} | "
            f"mean within-group source Jaccard: {fmt(statistics.mean(within_jaccards) if within_jaccards else None)}"
        ),
        "",
        "| group | n | source jaccard | answer chars mean | stdev |",
        "|---|---|---|---|---|",
    ]
    lines += [
        f"| {label[:60]} | {n} | {fmt(jac)} | {mean_chars} | {fmt(stdev)} |"
        for label, n, jac, mean_chars, stdev in consistency_rows
    ]
    lines += ["", "## Anomalies", ""]
    lines += [f"- {a}" for a in anomalies] if anomalies else ["- none"]
    return "\n".join(lines) + "\n"


def dry_run(prompts: list[str], groups: dict[str, list[int]], input_path: str) -> None:
    sizes = [len(m) for m in groups.values()]
    print(f"input: {input_path}")
    print(
        f"prompts: {len(prompts)} | groups: {len(groups)} | "
        f"size distribution: {dict(sorted((s, sizes.count(s)) for s in set(sizes)))}"
    )
    for label, members in list(groups.items())[:3]:
        print(f"  group '{label[:60]}' -> {members}")
        for m in members:
            print(f"    [{m}] {prompts[m - 1][:80]}")
    print(f"results would be written to: {os.path.abspath(RESULTS_DIR)}/")
    print("chat endpoint:", CHAT_BASE, "| auth token:", "present" if SERVICE_TOKEN else "MISSING")
    print("qdrant:", QDRANT_URL, "collection:", QDRANT_COLLECTION)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", type=int, default=1, help="1-based prompt index to resume from")
    ap.add_argument("--limit", type=int, default=None, help="max prompts to process")
    ap.add_argument("--delay", type=float, default=0.5, help="seconds between prompts (rate limiting)")
    ap.add_argument("--dry-run", action="store_true", help="validate prompt loading and grouping without executing")
    ap.add_argument("--input", default=DEFAULT_INPUT, help=f"prompt source (default: {DEFAULT_INPUT})")
    args = ap.parse_args()

    prompts = load_prompts(args.input)
    groups = group_prompts(prompts)
    if args.dry_run:
        dry_run(prompts, groups, args.input)
        return
    store = run_eval(prompts, groups, args.input, args.start, args.limit, args.delay)
    report_path = store.path.removesuffix(".json") + "_report.md"
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(build_report(store))
    print("results:", store.path)
    print("report:", report_path)


if __name__ == "__main__":
    main()
