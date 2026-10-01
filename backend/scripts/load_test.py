"""Concurrent load test for /search against a running backend.

``--mode cold`` sends N distinct queries so every request runs the full pipeline
(encode + rerank + Qdrant), stressing inference/CPU; ``--mode hot`` sends the
same query every time, stressing I/O through the caches. Prints total time,
requests/sec and p50/p95/p99 latency in ms.

    python3 scripts/load_test.py --base http://localhost:8001 \
        --concurrency 32 --total 64 --mode cold --workers 8
"""
import argparse
import concurrent.futures as cf
import http.client
import statistics
import threading
import time
import urllib.parse
import urllib.request


class _KeepAliveHandler(urllib.request.HTTPHandler, urllib.request.HTTPSHandler):
    """Per-thread HTTP/1.1 keep-alive handler.

    urllib's default opener opens a fresh socket per request and forces
    ``Connection: close``. The loader runs every task through a fixed-size
    ThreadPoolExecutor, so one live connection per worker thread avoids connect
    latency and socket churn. This is the ONLY protocol opener registered (the
    OpenerDirector is built by hand, not via ``build_opener``), so its
    ``http_open``/``https_open`` are not shadowed by the default handlers.
    """

    _local = threading.local()

    def http_open(self, req):
        return self._open(req, "http")

    def https_open(self, req):
        return self._open(req, "https")

    def _open(self, req, scheme):
        netloc = req.host  # e.g. "localhost:8001"
        timeout = req.timeout or 120
        conn = self._conn(scheme, netloc, timeout)
        headers = dict(req.header_items())
        # Preserve the caller's header names verbatim; .title()-casing them would
        # mangle multi-word names like "Content-Type".
        headers["Connection"] = "keep-alive"
        try:
            return self._exchange(conn, req, headers)
        except (OSError, http.client.HTTPException):
            # Server dropped the keep-alive socket; close the dead one so we
            # don't leak it, then reconnect once on a fresh connection.
            conn.close()
            conn = self._new(scheme, netloc, timeout)
            self._set_conn(conn)
            return self._exchange(conn, req, headers)

    def _conn(self, scheme, netloc, timeout):
        key = (scheme, netloc)
        conn = getattr(self._local, "conn", None)
        if conn is None or getattr(self._local, "key", None) != key:
            conn = self._new(scheme, netloc, timeout)
            self._set_conn(conn, key)
        return conn

    def _set_conn(self, conn, key=None):
        self._local.conn = conn
        if key is not None:
            self._local.key = key

    def _new(self, scheme, netloc, timeout):
        if scheme == "https":
            return http.client.HTTPSConnection(netloc, timeout=timeout)
        return http.client.HTTPConnection(netloc, timeout=timeout)

    def _exchange(self, conn, req, headers):
        if conn.sock is None:
            conn.connect()
        conn.request(
            req.get_method(), req.selector, req.data, headers,
            encode_chunked=req.has_header("Transfer-encoding"),
        )
        r = conn.getresponse()
        # Keep the socket open for this thread's next request: neuter the
        # context-manager close() so the caller's `with` cannot tear it down.
        r.close = lambda: None
        return r


# Built by hand so the default HTTPHandler/HTTPSHandler are NOT also registered
# (build_opener would add them and let their http_open win over ours); the
# redirect/error processors keep 3xx and 4xx/5xx behaving like the normal opener.
_opener = urllib.request.OpenerDirector()
_opener.add_handler(_KeepAliveHandler())
_opener.add_handler(urllib.request.HTTPRedirectHandler())
_opener.add_handler(urllib.request.HTTPErrorProcessor())
# OpenerDirector needs an UnknownHandler to reject unsupported schemes cleanly.
_opener.add_handler(urllib.request.UnknownHandler())


# Cold queries are unique per request so every one triggers a full retrieval
# pass, and the run id keeps them distinct across invocations so a per-worker
# in-process TTLCache cannot serve them.
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

    `latency_ms` is the round-trip time on success and None on any failure, in
    which case `error` carries a short status string so the caller can record it
    and keep the run alive.
    """
    query = "top startup funding deals of 2024" if hot else cold_query(int(q), run_id)
    u = f"{url}/search?top_k=8&q=" + urllib.parse.quote(query)
    t0 = time.perf_counter()
    try:
        with _opener.open(u, timeout=120) as r:
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

    tasks = [str(i) for i in range(args.total)]
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
