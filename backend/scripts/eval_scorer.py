"""LLM-as-judge scorer for eval_runner.py output.

Reads the JSON produced by eval_runner.py and scores each result on
faithfulness, relevance, context precision, citation accuracy, recency,
and refusal correctness using a Gemini judge model. Results are appended
one-per-line to a JSONL log (crash-safe) and consolidated into a summary
JSON + markdown report after the run completes.

Prerequisites:
  - GEMINI_API_KEY set in backend/.env
  - eval_runner.py output JSON in eval_results/

    ./venv/bin/python scripts/eval_scorer.py eval_results/20260901T120000Z.json
    ./venv/bin/python scripts/eval_scorer.py eval_results/20260901T120000Z.json --start 1 --limit 5

Results land in eval_results/<timestamp>_scores.json and <timestamp>_scores_report.md.
"""

import argparse
import fcntl
import json
import os
import re
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime

from dotenv import load_dotenv

_BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(dotenv_path=os.path.join(_BACKEND_DIR, ".env"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_BASE_URL = os.getenv("GEMINI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.1-flash-lite")
RESULTS_DIR = os.getenv("EVAL_RESULTS_DIR", "eval_results")
REQUEST_TIMEOUT = 120
MAX_RETRIES = 2
RETRY_BACKOFF = 2.0

JUDGE_PROMPT = """You are a strict quality evaluator for a news RAG chatbot. You will be given a USER QUESTION, the RETRIEVED CONTEXT (news article excerpts with publish dates), and the GENERATED ANSWER. Score the answer on the following dimensions. Be strict — do not give benefit of the doubt.

USER QUESTION:
{question}

RETRIEVED CONTEXT:
{retrieved_chunks_with_dates_and_sources}

GENERATED ANSWER:
{answer}

Evaluate and return ONLY valid JSON in this exact schema:

{{
  "faithfulness": {{
    "score": <1-5>,
    "reasoning": "<one sentence>",
    "unsupported_claims": ["<list any claim in the answer NOT backed by the context, empty array if none>"]
  }},
  "answer_relevance": {{
    "score": <1-5>,
    "reasoning": "<does it actually answer what was asked>"
  }},
  "context_precision": {{
    "score": <1-5>,
    "reasoning": "<what fraction of retrieved context was actually relevant/used>"
  }},
  "citation_accuracy": {{
    "score": <1-5 or null if no citations expected>,
    "reasoning": "<do citations point to sources that actually support the claim they're attached to>"
  }},
  "recency_correctness": {{
    "score": <1-5 or null if not time-sensitive>,
    "reasoning": "<did it use the most recent relevant article, not stale info>"
  }},
  "refusal_correctness": {{
    "score": <1-5 or null if not applicable>,
    "reasoning": "<if this is an out-of-scope/unanswerable question, did the model correctly decline instead of fabricating>"
  }},
  "hallucination_detected": <true/false>,
  "overall_pass": <true/false, true only if faithfulness >= 4 AND no hallucination_detected>
}}

Scoring guide: 5 = fully correct/grounded, 3 = partially correct with minor issues, 1 = wrong or fabricated. If ANY claim in the answer cannot be traced to the retrieved context, faithfulness must be 3 or lower and hallucination_detected must be true."""


class ApiError(Exception):
    def __init__(self, status: int, body: str):
        super().__init__(f"HTTP {status}: {body[:500]}")
        self.status = status
        self.body = body


_MAX_BACKOFF_SECONDS = 60


def _is_retryable_judge_error(exc: Exception) -> bool:
    """Return True only for transient errors worth retrying."""
    if isinstance(exc, ApiError):
        return exc.status >= 500 or exc.status == 429
    return isinstance(exc, (urllib.error.URLError, TimeoutError, ConnectionError, ConnectionResetError, json.JSONDecodeError))


def _retry_after_seconds(exc: Exception) -> float | None:
    """Parse Retry-After header from an ApiError (429), capped at 60s."""
    if not isinstance(exc, ApiError) or exc.status != 429:
        return None
    match = re.search(r"Retry-After:\s*(\d+)", exc.body, re.IGNORECASE)
    if not match:
        return None
    try:
        return min(float(match.group(1)), _MAX_BACKOFF_SECONDS)
    except ValueError:
        return None


def _req(url: str, payload: dict, timeout: int = REQUEST_TIMEOUT) -> dict:
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    if GEMINI_API_KEY:
        headers["Authorization"] = f"Bearer {GEMINI_API_KEY}"
    req = urllib.request.Request(url, data=data, method="POST", headers=headers)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(req, timeout=timeout) as resp:
            return json.load(resp)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise ApiError(exc.code, body) from exc


def _format_context(result: dict) -> str:
    """Build the retrieved context string from sources + bodies."""
    parts = []
    sources = result.get("sources") or []
    bodies = result.get("bodies") or {}
    for src in sources:
        sid = src.get("id")
        title = src.get("title", "Untitled")
        pub_date = src.get("published_at", src.get("publish_date", "unknown date"))
        source_name = src.get("source", "unknown")
        body = bodies.get(str(sid), "")[:2000]
        parts.append(
            f"[Article {sid}] \"{title}\" ({source_name}, {pub_date})\n{body}\n"
        )
    return "\n".join(parts) if parts else "(no context retrieved)"


def _call_judge(prompt: str) -> dict | None:
    """Call Gemini and return parsed JSON, or None on failure."""
    for attempt in range(MAX_RETRIES + 1):
        try:
            resp = _req(
                f"{GEMINI_BASE_URL.rstrip('/')}/chat/completions",
                {
                    "model": GEMINI_MODEL,
                    "max_tokens": 1500,
                    "messages": [{"role": "user", "content": prompt}],
                    "temperature": 0.0,
                },
            )
            content = resp["choices"][0]["message"]["content"]
            return _extract_json(content)
        except Exception as exc:
            if _is_retryable_judge_error(exc) and attempt < MAX_RETRIES:
                delay = _retry_after_seconds(exc)
                if delay is None:
                    delay = RETRY_BACKOFF * (2 ** attempt)
                print(f"  WARN: judge call failed ({type(exc).__name__}); retrying in {delay:.1f}s", flush=True)
                time.sleep(delay)
            else:
                print(f"  ERROR: judge call failed ({type(exc).__name__}): {exc}", flush=True)
                return None


def _extract_json(text: str) -> dict | None:
    """Extract the first JSON object from the LLM response."""
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def _score_entry(result: dict) -> dict:
    """Score a single eval_runner result."""
    if not result.get("ok") or not result.get("answer"):
        return {
            "index": result.get("index"),
            "error": result.get("error", "no answer"),
            "faithfulness": None,
            "answer_relevance": None,
            "context_precision": None,
            "citation_accuracy": None,
            "recency_correctness": None,
            "refusal_correctness": None,
            "hallucination_detected": None,
            "overall_pass": None,
        }

    context_str = _format_context(result)
    judge_prompt = JUDGE_PROMPT.format(
        question=result["prompt"],
        retrieved_chunks_with_dates_and_sources=context_str,
        answer=result["answer"],
    )
    scores = _call_judge(judge_prompt)
    if scores is None:
        return {
            "index": result.get("index"),
            "error": "judge call failed",
            "faithfulness": None,
            "answer_relevance": None,
            "context_precision": None,
            "citation_accuracy": None,
            "recency_correctness": None,
            "refusal_correctness": None,
            "hallucination_detected": None,
            "overall_pass": None,
        }

    return {
        "index": result.get("index"),
        "group": result.get("group"),
        "prompt": result["prompt"][:200],
        "error": None,
        "faithfulness": scores.get("faithfulness"),
        "answer_relevance": scores.get("answer_relevance"),
        "context_precision": scores.get("context_precision"),
        "citation_accuracy": scores.get("citation_accuracy"),
        "recency_correctness": scores.get("recency_correctness"),
        "refusal_correctness": scores.get("refusal_correctness"),
        "hallucination_detected": scores.get("hallucination_detected"),
        "overall_pass": scores.get("overall_pass"),
    }


class ScoreStore:
    """Crash-safe JSONL append for scored results."""

    def __init__(self, json_path: str):
        self.json_path = json_path
        self.log_path = json_path.removesuffix(".json") + ".jsonl"
        self.scores: list[dict] = []
        os.makedirs(os.path.dirname(json_path) or ".", exist_ok=True)
        self._log_fh = open(self.log_path, "a", encoding="utf-8")
        try:
            fcntl.flock(self._log_fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._log_fh.close()
            raise SystemExit(f"another scorer run holds the lock on {self.log_path}; aborting") from None

    def add(self, entry: dict) -> None:
        self.scores.append(entry)
        self._log_fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self._log_fh.flush()
        os.fsync(self._log_fh.fileno())

    def finish(self, meta: dict) -> dict:
        doc = {"meta": meta, "scores": self.scores}
        tmp = self.json_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=1, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.json_path)
        return doc

    def cleanup_log(self) -> None:
        self._log_fh.close()
        os.remove(self.log_path)


def _build_report(meta: dict, scores: list[dict]) -> str:
    """Generate markdown summary of scoring results."""
    scored = [s for s in scores if s.get("error") is None]
    errored = [s for s in scores if s.get("error") is not None]

    def _avg(key: str) -> float | None:
        vals = [
            s[key]["score"]
            for s in scored
            if isinstance(s.get(key), dict) and isinstance(s[key].get("score"), (int, float))
        ]
        return sum(vals) / len(vals) if vals else None

    def _hallucination_rate() -> float | None:
        vals = [s for s in scored if isinstance(s.get("hallucination_detected"), bool)]
        if not vals:
            return None
        return sum(1 for v in vals if v) / len(vals)

    def _pass_rate() -> float | None:
        vals = [s for s in scored if isinstance(s.get("overall_pass"), bool)]
        if not vals:
            return None
        return sum(1 for v in vals if v) / len(vals)

    def _fmt(v) -> str:
        return "n/a" if v is None else f"{v:.2f}"

    dimensions = [
        ("faithfulness", _avg("faithfulness")),
        ("answer_relevance", _avg("answer_relevance")),
        ("context_precision", _avg("context_precision")),
        ("citation_accuracy", _avg("citation_accuracy")),
        ("recency_correctness", _avg("recency_correctness")),
        ("refusal_correctness", _avg("refusal_correctness")),
    ]

    lines = [
        f"# Eval scores report — {meta.get('started_utc', 'unknown')}",
        "",
        f"- Source: `{meta.get('source_file')}`",
        f"- Judge model: `{meta.get('judge_model')}`",
        f"- Total results: {len(scores)} | scored: {len(scored)} | errors: {len(errored)}",
        "",
        "## Aggregate scores (1–5 scale)",
        "",
        "| dimension | mean |",
        "|---|---|",
    ]
    for dim, val in dimensions:
        lines.append(f"| {dim} | {_fmt(val)} |")

    lines += [
        "",
        f"- Hallucination rate: {_fmt(_hallucination_rate())}",
        f"- Overall pass rate: {_fmt(_pass_rate())}",
        "",
    ]

    if errored:
        lines.append("## Errors")
        lines.append("")
        for e in errored:
            lines.append(f"- #{e.get('index')}: {e.get('error')}")
        lines.append("")

    per_group: dict[str, list[dict]] = {}
    for s in scored:
        per_group.setdefault(s.get("group") or "?", []).append(s)
    if len(per_group) > 1:
        lines.append("## Per-group scores")
        lines.append("")
        lines.append("| group | n | faithfulness | hallucination rate | pass rate |")
        lines.append("|---|---|---|---|---|")
        for g, members in sorted(per_group.items()):
            f_vals = [
                m["faithfulness"]["score"]
                for m in members
                if isinstance(m.get("faithfulness"), dict) and isinstance(m["faithfulness"].get("score"), (int, float))
            ]
            f_avg = sum(f_vals) / len(f_vals) if f_vals else None
            h_vals = [m for m in members if isinstance(m.get("hallucination_detected"), bool)]
            h_rate = sum(1 for v in h_vals if v) / len(h_vals) if h_vals else None
            p_vals = [m for m in members if isinstance(m.get("overall_pass"), bool)]
            p_rate = sum(1 for v in p_vals if v) / len(p_vals) if p_vals else None
            lines.append(f"| {g[:50]} | {len(members)} | {_fmt(f_avg)} | {_fmt(h_rate)} | {_fmt(p_rate)} |")
        lines.append("")

    return "\n".join(lines) + "\n"


def main() -> None:
    if not GEMINI_API_KEY:
        raise SystemExit("GEMINI_API_KEY missing from backend/.env — cannot run judge.")

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input_json", help="Path to eval_runner output JSON")
    ap.add_argument("--start", type=int, default=1, help="1-based index to resume from")
    ap.add_argument("--limit", type=int, default=None, help="max results to score")
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between judge calls (rate limiting)")
    args = ap.parse_args()

    with open(args.input_json, encoding="utf-8") as fh:
        doc = json.load(fh)
    results = doc.get("results", [])
    source_meta = doc.get("meta", {})

    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    out_dir = os.path.dirname(args.input_json) or RESULTS_DIR
    os.makedirs(out_dir, exist_ok=True)
    store = ScoreStore(os.path.join(out_dir, f"{ts}_scores.json"))

    meta = {
        "source_file": os.path.basename(args.input_json),
        "source_started_utc": source_meta.get("started_utc"),
        "judge_model": GEMINI_MODEL,
        "started_utc": ts,
        "total_results": len(results),
        "resumed_from": args.start if args.start > 1 else None,
    }

    total = len(results)
    for i, result in enumerate(results, start=1):
        if i < args.start:
            continue
        if args.limit is not None and i >= args.start + args.limit:
            break

        score = _score_entry(result)
        store.add(score)

        status = "ok" if score.get("error") is None else f"ERR {score['error']}"
        faith = score.get("faithfulness", {})
        faith_s = faith.get("score") if isinstance(faith, dict) else "n/a"
        hall = score.get("hallucination_detected")
        print(
            f"[{i:>3}/{total}] {status} faith={faith_s} halluc={hall} :: {result.get('prompt', '')[:60]}",
            flush=True,
        )
        time.sleep(args.delay)

    doc_out = store.finish(meta)
    report_path = store.json_path.removesuffix(".json") + "_report.md"
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write(_build_report(meta, doc_out["scores"]))
        fh.flush()
        os.fsync(fh.fileno())
    store.cleanup_log()
    print("scores:", store.json_path)
    print("report:", report_path)


if __name__ == "__main__":
    main()
