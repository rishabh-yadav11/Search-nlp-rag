"""Unified chat eval runner over prompts_with_variations_flat.json.

For every prompt: create a chat session, POST one message, capture the answer,
sources, token usage and cost, fetch the cited articles' full bodies from
Qdrant, then delete the session. Results are appended one-per-line to a JSONL
log after every prompt so an interrupted run keeps partial data; pass
--start/--limit to resume. A companion markdown report aggregates
success/latency/cost, cross-variation consistency (prompts are grouped by
normalized text similarity) and anomalies.

Prerequisites: backend on localhost:8001 with AUTH_SERVICE_TOKEN in
backend/.env, plus Qdrant reachable at QDRANT_URL. Chat turns make real (billed)
LLM calls against the daily budget — use --dry-run to validate loading only.

    ./venv/bin/python scripts/eval_runner.py --dry-run
    ./venv/bin/python scripts/eval_runner.py --start 1 --limit 5
    ./venv/bin/python scripts/eval_runner.py --input /path/to/prompts.json

Results land in eval_results/<timestamp>.json and <timestamp>_report.md,
relative to the working directory (override with EVAL_RESULTS_DIR). Each
invocation writes its own file covering the slice it ran; a resumed slice's
report aggregates only that slice. Run one eval at a time — concurrent runs
would collide on filenames. During the run results are appended one-per-line
to <timestamp>.jsonl (crash-safe, cheap) under an exclusive advisory lock;
after the report is written the .jsonl is removed. If a run dies before that
(SIGKILL, power loss), the OS releases the lock and the next invocation
consolidates the orphaned .jsonl — including a report — before starting
fresh. Recovery never touches a live run's log: the lock attempt blocks it.
"""

import argparse
import fcntl
import json
import math
import os
import re
import signal
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
            data = json.load(resp)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise ApiError(exc.code, body) from exc
    if not isinstance(data, dict):
        raise ValueError(f"expected a JSON object response, got {type(data).__name__}")  # noqa: TRY004 — API-shape guard, not a local type bug
    return data


def _delete(url: str) -> None:
    """DELETE that never fails on a 2xx response with an empty/non-object JSON body."""
    try:
        _req(url, method="DELETE")
    except (json.JSONDecodeError, ValueError):
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
        base = min((keys[m - 1] for m in members), key=lambda k: (len(k), k))
        label, n = base, 2
        while label in grouped:  # identical normalized text can occur in separate runs
            label = f"{base} ({n})"
            n += 1
        grouped[label] = sorted(members)
    return dict(sorted(grouped.items(), key=lambda kv: kv[1][0]))


class ResultStore:
    """Appends one JSON line per prompt so a long run can be interrupted at any
    point without losing earlier results, and never re-serializes the growing
    document. An exclusive advisory lock is held on the log for the store's
    lifetime: recovery uses lock acquisition to tell a dead run's log from a
    live one. finish() consolidates the log into the single <ts>.json document
    the report and downstream tooling expect; cleanup_log() releases the lock
    and removes the log once the report is safely on disk."""

    def __init__(self, json_path: str, meta: dict):
        self.json_path = json_path
        self.log_path = json_path.removesuffix(".json") + ".jsonl"
        self.meta = meta
        self.results: list[dict] = []
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
        self._log_fh = open(self.log_path, "a", encoding="utf-8")
        fcntl.flock(self._log_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        self._log_fh.write(json.dumps({"meta": meta}, ensure_ascii=False) + "\n")
        self._log_fh.flush()

    def add(self, entry: dict) -> None:
        self.results.append(entry)
        self._log_fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._log_fh.flush()

    def log_meta_update(self, update: dict) -> None:
        """Persist a mid-run meta mutation so recovery of a dead run's log
        reconstructs the final meta (only result entries carry "index")."""
        self._log_fh.write(json.dumps(update, ensure_ascii=False) + "\n")
        self._log_fh.flush()

    def finish(self) -> dict:
        doc = {"meta": self.meta, "results": self.results}
        tmp = self.json_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=1, ensure_ascii=False)
        os.replace(tmp, self.json_path)
        return doc

    def cleanup_log(self) -> None:
        self._log_fh.close()
        os.remove(self.log_path)


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
        fetched = {}
        for p in points:
            body = (p.payload or {}).get("body")
            if isinstance(body, str) and body:
                fetched[str(p.id)] = body
        return fetched, None
    except Exception as exc:
        return {}, repr(exc)


def run_prompt(client: "QdrantClient", index: int, group: str | None, prompt: str, sid: str | int) -> dict:
    t0 = time.perf_counter()
    try:
        d = _req(CHAT_BASE + f"/sessions/{sid}/messages", {"content": prompt})
    except Exception as exc:
        return _failure_entry(index, group, prompt, repr(exc), session_id=sid)
    latency_ms = round((time.perf_counter() - t0) * 1000)
    a = d.get("assistant")
    if not isinstance(a, dict):
        a = {}
    raw_content = a.get("content")
    answer = raw_content if isinstance(raw_content, str) else None
    sources = a.get("sources")
    sources = sources if isinstance(sources, list) else []
    source_ids = _citation_ids(sources)
    dropped_citations = sum(1 for s in sources if not (isinstance(s, dict) and isinstance(s.get("id"), int)))
    bodies, body_error = _fetch_bodies(client, source_ids) if source_ids else ({}, None)
    return {
        "index": index,
        "group": group,
        "prompt": prompt,
        "session_id": sid,
        "ok": answer is not None,
        "error": None if answer is not None else "assistant content missing or not a string",
        "raw_response": None if answer is not None else d,
        "answer": answer,
        "sources": sources,
        "bodies": bodies,
        "body_error": body_error,
        "n_sources": len(sources),
        "n_cited_ids": len(source_ids),
        "n_dropped_citations": dropped_citations,
        "prompt_tokens": a.get("prompt_tokens"),
        "completion_tokens": a.get("completion_tokens"),
        "cost_inr": a.get("cost"),
        "latency_ms": latency_ms,
        "note": d.get("note") or a.get("note"),
    }


def _recover_orphan_logs(results_dir: str) -> None:
    """Consolidate any <ts>.jsonl left behind by a run that died before its
    report was written (SIGKILL, power loss, hard crash) so its completed
    prompts are not stranded in a log nothing reads. A log still held by a
    live run is skipped: the advisory lock attempt fails. Tolerates a torn
    final line."""
    if not os.path.isdir(results_dir):
        return
    for log_name in sorted(os.listdir(results_dir)):
        if not log_name.endswith(".jsonl"):
            continue
        log_path = os.path.join(results_dir, log_name)
        json_path = log_path.removesuffix(".jsonl") + ".json"
        report_path = json_path.removesuffix(".json") + "_report.md"
        try:
            lock_fh = open(log_path, "a", encoding="utf-8")
        except OSError as exc:
            print(f"WARN: cannot open eval log {log_path}: {exc!r}", flush=True)
            continue
        try:
            try:
                fcntl.flock(lock_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                print(f"skipping {log_path}: locked — a run is active", flush=True)
                continue
            if os.path.exists(json_path) and os.path.exists(report_path):
                os.remove(log_path)
                continue
            with open(log_path, encoding="utf-8") as fh:
                first_line, *rest = fh.readlines()
            meta = json.loads(first_line).get("meta")
            if not isinstance(meta, dict):
                raise ValueError("first line carries no meta record")  # noqa: TRY004 — data-shape guard, not a local type bug
            results = []
            for line in rest:
                try:
                    record = json.loads(line)
                except ValueError:
                    print(f"WARN: dropped a truncated result line from {log_path}", flush=True)
                    continue
                if isinstance(record, dict) and "index" not in record:
                    if isinstance(record.get("orphan_session_risk"), int):
                        meta["orphan_session_risk"] = record["orphan_session_risk"]
                    continue
                results.append(record)
            tmp = json_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({"meta": meta, "results": results}, fh, indent=1, ensure_ascii=False)
            os.replace(tmp, json_path)
            with open(report_path, "w", encoding="utf-8") as fh:
                fh.write(build_report(meta, results))
            os.remove(log_path)
            print(f"recovered {len(results)} result(s) from orphaned log {log_path}", flush=True)
        except (OSError, ValueError, IndexError) as exc:
            print(f"WARN: skipping unreadable eval log {log_path}: {exc!r}", flush=True)
        finally:
            lock_fh.close()


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

    _recover_orphan_logs(RESULTS_DIR)
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
    resume_at = start
    interrupted = False

    def _terminate(signum, frame) -> None:
        raise KeyboardInterrupt

    previous_term_handler = signal.signal(signal.SIGTERM, _terminate)
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
                if sid is None:
                    entry = _failure_entry(index, group, prompt, "session create response missing id")
                else:
                    try:
                        entry = run_prompt(qdrant, index, group, prompt, sid)
                    except Exception as exc:
                        entry = _failure_entry(index, group, prompt, f"message phase failed: {exc!r}", session_id=sid)
            finally:
                if sid is not None:
                    try:
                        _delete(CHAT_BASE + f"/sessions/{sid}")
                    except Exception:
                        meta["orphan_session_risk"] += 1
                        store.log_meta_update({"orphan_session_risk": meta["orphan_session_risk"]})
                        print("WARN: failed to delete eval chat session (orphan risk)", flush=True)
                    except BaseException:
                        meta["orphan_session_risk"] += 1
                        store.log_meta_update({"orphan_session_risk": meta["orphan_session_risk"]})
                        print("WARN: eval chat session left orphaned by interrupt", flush=True)
                        raise
            store.add(entry)
            resume_at = index + 1
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
        signal.signal(signal.SIGTERM, previous_term_handler)
        qdrant.close()
    if interrupted:
        print(f"interrupted before prompt {resume_at}; resume with --start {resume_at}", flush=True)
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


def build_report(meta: dict, results: list[dict]) -> str:
    ok = [r for r in results if r.get("ok")]
    latencies = [r["latency_ms"] for r in ok if isinstance(r.get("latency_ms"), (int, float))]
    costs = [r["cost_inr"] for r in ok if isinstance(r.get("cost_inr"), (int, float)) and r["cost_inr"]]
    median_cost = statistics.median(costs) if costs else None

    def _num(value) -> int:
        return value if isinstance(value, (int, float)) else 0

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
        dropped = r.get("n_dropped_citations") or 0
        if dropped:
            anomalies.append(f"{label}: {dropped} source citation(s) without a parsable article id")
        missing_bodies = r.get("n_cited_ids", 0) - len(r.get("bodies") or {})
        if missing_bodies > 0:
            anomalies.append(f"{label}: {missing_bodies} cited source(s) without body text")
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
            f"- Total cost: {sum(costs):.2f} INR | prompt tokens: {sum(_num(r.get('prompt_tokens')) for r in ok)}, "
            f"completion tokens: {sum(_num(r.get('completion_tokens')) for r in ok)}"
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
    if args.delay < 0 or not math.isfinite(args.delay):
        ap.error("--delay must be a finite number >= 0")
    if args.start < 1:
        ap.error("--start must be >= 1")
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be >= 1")

    prompts = load_prompts(args.input)
    groups = group_prompts(prompts)
    if args.dry_run:
        dry_run(prompts, groups, args.input)
        return
    store = run_eval(prompts, groups, args.input, args.start, args.limit, args.delay)
    doc = store.finish()
    report_path = store.json_path.removesuffix(".json") + "_report.md"
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(build_report(doc["meta"], doc["results"]))
    store.cleanup_log()
    print("results:", store.json_path)
    print("report:", report_path)


if __name__ == "__main__":
    main()
