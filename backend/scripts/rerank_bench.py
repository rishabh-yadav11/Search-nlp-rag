"""Cross-encoder reranker benchmark: bge-reranker-base (torch / ONNX fp32 /
ONNX int8) vs the current ms-marco-MiniLM-L-6-v2 (torch).

Run on the deployment box from `backend/` with the venv python:

    ./venv/bin/python scripts/rerank_bench.py

Prerequisites: Qdrant reachable (config from backend/.env), the
`vccircle_articles` collection populated. Deterministic: candidates are built
from a fixed query list with a fixed seed, so re-runs are comparable.

Measures, per backend:
  * latency (median ms) for a realistic batch of RERANK_CANDIDATES pairs and
    for single pairs, under the same thread budget as production (TORCH_THREADS)
  * ranking agreement (Spearman rho) and top-8 overlap against every other
    backend, computed on the exact same candidate pairs
  * on-disk model size

The point: decide whether INT8 dynamic quantization of bge-reranker-base is
fast enough without meaningfully changing top-8 ordering.
"""
import asyncio
import os
import statistics
import sys
import time

sys.path.insert(0, os.path.abspath("."))

from app.config import config
from app.encoders import DenseEncoder

BGE = "BAAI/bge-reranker-base"
MINILM = "cross-encoder/ms-marco-MiniLM-L-6-v2"
BENCH_DIR = os.path.join("data", "reranker_bench")

QUERIES = [
    "How did demonetisation in 2016 impact fintech startups in India?",
    "Top startup funding deals of 2024",
    "How did Ola Cabs raise funding from SoftBank?",
    "What are the biggest venture capital deals in fintech this year?",
    "What happened to edtech startups during the 2020 Covid-19 lockdown?",
    "Which companies presented at the Techcircle DEMO India 2013 event?",
    "How did the 2008 global financial crisis change the way the Reserve Bank of India communicates?",
    "List the top 5 biggest funding rounds in India in 2025",
    "How many venture capital deals happened in India in 2024?",
    "What is the latest news about AI startups raising funding in India?",
    "Which game providers are listed as partners of Ninewins at non-GamStop betting sites?",
    "Ask Property Fund exits Shriram Properties project",
    "Unicorns created in India last year",
    "Venture debt providers in India",
    "Manufacturing companies that raised Series B funding",
]

THREADS = config.TORCH_THREADS


# ---------------------------------------------------------------------------
# backend factories
# ---------------------------------------------------------------------------

def _session(path: str, threads: int):
    import onnxruntime as ort

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    return ort.InferenceSession(
        path, sess_options=opts, providers=["CPUExecutionProvider"]
    )


class Backend:
    def __init__(self, name: str, predict, size_mb: float):
        self.name = name
        self.predict = predict
        self.size_mb = size_mb

    def score(self, pairs):
        return [float(s) for s in self.predict(pairs)]


def make_torch(model_name: str, threads: int):
    from sentence_transformers import CrossEncoder

    m = CrossEncoder(model_name, device="cpu", max_length=512)
    return lambda pairs: m.predict(pairs, show_progress_bar=False, batch_size=64)


def export_onnx(model_name: str, tag: str):
    """Export the model to ONNX (fp32) under BENCH_DIR/<tag>/; returns path."""
    from optimum.onnxruntime import ORTModelForSequenceClassification
    from transformers import AutoTokenizer

    out = os.path.join(BENCH_DIR, tag)
    if os.path.isfile(os.path.join(out, "model.onnx")):
        return out
    os.makedirs(out, exist_ok=True)
    model = ORTModelForSequenceClassification.from_pretrained(model_name, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    return out


def quantize_onnx(fp32_dir: str, tag: str):
    """Dynamic INT8 quantize the fp32 export under BENCH_DIR/<tag>/."""
    from optimum.onnxruntime import ORTQuantizer
    from optimum.onnxruntime.configuration import AutoQuantizationConfig

    out = os.path.join(BENCH_DIR, tag)
    if os.path.isfile(os.path.join(out, "model.onnx")):
        return out
    os.makedirs(out, exist_ok=True)
    quantizer = ORTQuantizer.from_pretrained(fp32_dir, file_name="model.onnx")
    dqconfig = AutoQuantizationConfig.avx512(is_static=False)
    quantizer.quantize(save_dir=out, quantization_config=dqconfig)
    for f in ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.txt"):
        src = os.path.join(fp32_dir, f)
        if os.path.isfile(src) and not os.path.exists(os.path.join(out, f)):
            import shutil

            shutil.copy2(src, os.path.join(out, f))
    return out


def make_onnx(dir_path: str, threads: int):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(dir_path)
    sess = _session(os.path.join(dir_path, "model.onnx"), threads)

    def predict(pairs):
        inputs = tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")
        feed = {k: v.numpy() for k, v in inputs.items()}
        logits = sess.run(None, feed)[0]
        if logits.ndim == 2:
            return logits[:, 0].tolist()
        return logits.tolist()

    return predict


def dir_mb(path: str) -> float:
    total = 0.0
    for root, _dirs, files in os.walk(path):
        for f in files:
            total += os.path.getsize(os.path.join(root, f))
    return round(total / 1e6, 1)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

def _ranks(vals: list[float]) -> list[float]:
    """Standard competition ranking."""
    order = sorted(range(len(vals)), key=lambda i: vals[i], reverse=True)
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        r = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = r
        i = j + 1
    return ranks


def spearman(a: list[float], b: list[float]) -> float:
    if len(a) < 3:
        return 0.0
    ra, rb = _ranks(a), _ranks(b)
    n = len(a)
    d = [x - y for x, y in zip(ra, rb)]
    denom = (n ** 3 - n) / 6
    if denom == 0:
        return 0.0
    return 1.0 - 6 * sum(x * x for x in d) / denom


def topk_overlap(a: list[float], b: list[float], k: int = 8) -> float:
    sa = {i for i, _ in sorted(enumerate(a), key=lambda t: t[1], reverse=True)[:k]}
    sb = {i for i, _ in sorted(enumerate(b), key=lambda t: t[1], reverse=True)[:k]}
    return len(sa & sb) / k


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    print(f"threads={THREADS} queries={len(QUERIES)} candidates={config.RERANK_CANDIDATES}")
    print(f"torch threads env: OMP={os.environ.get('OMP_NUM_THREADS')} MKL={os.environ.get('MKL_NUM_THREADS')}")

    # -- build deterministic candidate sets from the live collection ----------
    # The app state isn't running here, so drive retrieval with a local
    # client + encoders instead of importing app.state.
    from qdrant_client import AsyncQdrantClient

    async def _candidates():
        dense = DenseEncoder(config.EMBED_MODEL, config.EMBED_DEVICE, config.TORCH_THREADS)
        from fastembed import SparseTextEmbedding

        sparse_m = SparseTextEmbedding(config.SPARSE_MODEL)
        client = AsyncQdrantClient(url=config.QDRANT_URL, timeout=30)
        try:
            groups = []
            for q in QUERIES:
                async with __import__("app.main", fromlist=["inference_lock"]).inference_lock:
                    dense_vec = (await asyncio.to_thread(dense.encode, q)).tolist()
                    sp = next(iter(sparse_m.embed([q])))
                from qdrant_client.models import Fusion, FusionQuery, Prefetch, SparseVector

                sparse_vec = SparseVector(indices=sp.indices.tolist(), values=sp.values.tolist())
                r = await client.query_points(
                    collection_name=config.QDRANT_COLLECTION,
                    prefetch=[
                        Prefetch(query=dense_vec, using="dense", limit=config.RERANK_CANDIDATES * 4),
                        Prefetch(query=sparse_vec, using="sparse", limit=config.RERANK_CANDIDATES * 4),
                    ],
                    query=FusionQuery(fusion=Fusion.RRF),
                    limit=config.RERANK_CANDIDATES,
                    with_payload=["title", "summary"],
                )
                pairs = [
                    (q, f"{p.payload.get('title', '')}. {p.payload.get('summary') or ''}".strip())
                    for p in r.points
                ]
                groups.append(pairs)
            return groups
        finally:
            await client.close()

    pair_sets = asyncio.run(_candidates())
    assert all(len(s) == config.RERANK_CANDIDATES for s in pair_sets), "candidate sets incomplete"
    print(f"candidate sets: {len(pair_sets)} x {config.RERANK_CANDIDATES} pairs")

    # -- build backends --------------------------------------------------------
    bge_fp32_dir = export_onnx(BGE, "bge_reranker_base_fp32")
    bge_int8_dir = quantize_onnx(bge_fp32_dir, "bge_reranker_base_int8")
    backends = [
        Backend("bge  torch", make_torch(BGE, THREADS), size_mb=0.0),
        Backend("bge  onnx-fp32", make_onnx(bge_fp32_dir, THREADS), size_mb=dir_mb(bge_fp32_dir)),
        Backend("bge  onnx-int8 ", make_onnx(bge_int8_dir, THREADS), size_mb=dir_mb(bge_int8_dir)),
        Backend("MiniLM torch", make_torch(MINILM, THREADS), size_mb=0.0),
    ]

    # -- score + latency --------------------------------------------------------
    scores = {b.name: [] for b in backends}
    lat_batch: dict[str, list[float]] = {b.name: [] for b in backends}
    lat_single: dict[str, list[float]] = {b.name: [] for b in backends}

    for q, pairs in zip(QUERIES, pair_sets):
        for b in backends:
            t0 = time.perf_counter()
            s = b.score(pairs)
            lat_batch[b.name].append((time.perf_counter() - t0) * 1000)
            scores[b.name].extend(s)
            t0 = time.perf_counter()
            for p in pairs:
                b.score([p])
            lat_single[b.name].append((time.perf_counter() - t0) * 1000 / len(pairs))

    # -- per-query agreement -----------------------------------------------------
    names = [b.name for b in backends]
    rho = {n: {m: [] for m in names} for n in names}
    ovl = {n: {m: [] for m in names} for n in names}
    nq = len(pair_sets)
    for qi in range(nq):
        q_scores = {n: scores[n][qi * config.RERANK_CANDIDATES:(qi + 1) * config.RERANK_CANDIDATES] for n in names}
        for a in names:
            for b in names:
                rho[a][b].append(spearman(q_scores[a], q_scores[b]))
                ovl[a][b].append(topk_overlap(q_scores[a], q_scores[b]))

    print("\n=== latency (median ms) ===")
    print(f"{'backend':<16}{'batch12':>10}{'single':>10}{'sizeMB':>9}")
    for b in backends:
        size = f"{b.size_mb:.1f}" if b.size_mb > 0 else "-"
        print(f"{b.name:<16}{statistics.median(lat_batch[b.name]):>10.1f}"
              f"{statistics.median(lat_single[b.name]):>10.1f}{size:>9}")

    print("\n=== Spearman rho (mean across queries) ===")
    print(f"{'':>16}" + "".join(f"{n:>16}" for n in names))
    for a in names:
        print(f"{a:>16}" + "".join(f"{statistics.mean(rho[a][b]):>16.3f}" for b in names))

    print("\n=== top-8 overlap (mean across queries) ===")
    print(f"{'':>16}" + "".join(f"{n:>16}" for n in names))
    for a in names:
        print(f"{a:>16}" + "".join(f"{statistics.mean(ovl[a][b]):>16.2f}" for b in names))

    int8_vs_fp32 = statistics.mean(ovl["bge  onnx-int8 "]["bge  onnx-fp32"])
    speedup = statistics.median(lat_batch["bge  onnx-fp32"]) / statistics.median(lat_batch["bge  onnx-int8 "])
    print(f"\nint8 vs fp32: top-8 overlap={int8_vs_fp32:.3f}  batch speedup={speedup:.2f}x")
    ok = int8_vs_fp32 >= 0.98 and speedup >= 1.4
    print(f"recommend {'int8' if ok else 'fp32'} (overlap>=0.98 and speedup>=1.4: {'YES' if ok else 'NO'})")


if __name__ == "__main__":
    main()
