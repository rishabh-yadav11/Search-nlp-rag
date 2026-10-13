
import argparse
import concurrent.futures as cf
import http.client
import statistics
import threading
import time
import urllib.parse
import urllib.request


class _KeepAliveHandler(urllib.request.HTTPHandler, urllib.request.HTTPSHandler):


    _local = threading.local()

    def http_open(self, req):
        return self._open(req, "http")

    def https_open(self, req):
        return self._open(req, "https")

    def _open(self, req, scheme):
        netloc = req.host
        timeout = req.timeout or 120
        conn = self._conn(scheme, netloc, timeout)
        headers = dict(req.header_items())


        headers["Connection"] = "keep-alive"
        try:
            return self._exchange(conn, req, headers)
        except (OSError, http.client.HTTPException):


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


        r.close = lambda: None
        return r




_opener = urllib.request.OpenerDirector()
_opener.add_handler(_KeepAliveHandler())
_opener.add_handler(urllib.request.HTTPRedirectHandler())
_opener.add_handler(urllib.request.HTTPErrorProcessor())

_opener.add_handler(urllib.request.UnknownHandler())





def cold_query(i: int, run_id: int) -> str:
    topics = [
        "venture debt providers", "fintech funding round", "AI startups raising capital",
        "electric vehicle charging companies", "edtech deals 2023", "crypto exchange funding",
        "healthcare private equity India", "manufacturing series B", "saas companies growth",
        "unicorn creation 2025",
    ]
    return f"{topics[i % len(topics)]} {run_id}-{i}"


def hit(url: str, q: str, hot: bool, run_id: int):

    query = "top startup funding deals of 2024" if hot else cold_query(int(q), run_id)
    u = f"{url}/search?top_k=8&q=" + urllib.parse.quote(query)
    t0 = time.perf_counter()
    try:
        with _opener.open(u, timeout=120) as r:
            r.read()
        return (time.perf_counter() - t0) * 1000, None
    except urllib.error.HTTPError as e:
        return None, f"HTTP {e.code}"
    except Exception as e:
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
