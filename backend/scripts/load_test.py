"""Simple concurrent load test for /search.

Measures throughput and latency under concurrency against a running backend.
Two modes:

  cold  - N DISTINCT queries so every request hits the full pipeline
          (encode + rerank + Qdrant); stresses inference/CPU.
  hot   - every request uses the SAME query (cache hits); stresses I/O.

Run from anywhere (the backend must be up):

    python3 scripts/load_test.py --base http://localhost:8001 \
        --concurrency 32 --total 64 --mode cold --workers 8

Prints total time, requests/sec, and p50/p95/p99 latency (ms).
"""
import argparse
import concurrent.futures as cf
import statistics
import time
import urllib.parse
import urllib.request


# Cold queries: each request gets a UNIQUE query so every one triggers a full
# retrieval pass (distinct queries can't hit the search/retrieve cache). A
# per-run offset makes queries differ across invocations so the per-worker
# in-process TTLCache (not cleared by redis FLUSHDB) can't serve them.
def cold_query(i: int, run_id: int) -> str:
    topics = [
        "venture debt providers", "fintech funding round", "AI startups raising capital",
        "electric vehicle charging companies", "edtech deals 2023", "crypto exchange funding",
        "healthcare private equity India", "manufacturing series B", "saas companies growth",
        "unicorn creation 2025",
    ]
    return f"{topics[i % len(topics)]} {run_id}-{i}"


def hit(url: str, q: str, hot: bool, run_id: int):
    """Return (latency_ms, error).

    On a successful request ``error`` is None and ``latency_ms`` holds the
    measured round-trip time. On any failure (timeout, HTTP 429/500, connection
    reset, ...) ``latency_ms`` is None and ``error`` carries a short status
    string so the caller can record the failure and keep the run alive.
    """
    query = "top startup funding deals of 2024" if hot else cold_query(int(q), run_id)
    u = f"{url}/search?top_k=8&q=" + urllib.parse.quote(query)
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(u, timeout=120) as r:
            r.read()
        return (time.perf_counter() - t0) * 1000, None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}"
    except Exception as e:  # timeout / URLError / connection reset, etc.
        return None, type(e).__name__


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8001")
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--total", type=int, default=48)
    ap.add_argument("--mode", choices=["cold", "hot"], default="cold")
    ap.add_argument("--workers", type=int, default=None, help="label only")
    ap.add_argument("--run-id", type=int, default=0, help="cold-query namespace per run")
    args = ap.parse_args()

    tasks = [str(i % 10000) for i in range(args.total)]
    latencies: list[float] = []
    failures: list[str] = []
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = [ex.submit(hit, args.base, t, args.mode == "hot", args.run_id) for t in tasks]
        for f in cf.as_completed(futs):
            latency, err = f.result()
            if err is None:
                latencies.append(latency)
            else:
                failures.append(err)
    total_s = time.perf_counter() - t0

    latencies.sort()
    label = f"workers={args.workers or '?'}"
    if len(latencies) >= 2:
        p = lambda q: statistics.quantiles(latencies, n=100)[q - 1]
        pctl = f"p50={p(50):.0f}ms p95={p(95):.0f}ms p99={p(99):.0f}ms"
    else:
        pctl = "p50=- p95=- p99=- (need >=2 samples)"
    fail_str = f"  failures={len(failures)}"
    if failures:
        from collections import Counter
        fail_str += " " + " ".join(f"{k}:{v}" for k, v in Counter(failures).items())
    print(f"[{args.mode:>4}] {label}  total={args.total} conc={args.concurrency} "
          f"time={total_s:.2f}s  rps={args.total / total_s:.1f}  {pctl}{fail_str}")


if __name__ == "__main__":
    main()
