"""Cross-encoder reranker with an ONNX (optimum/onnxruntime) fast path and a
torch sentence-transformers fallback.

The ONNX path runs the MiniLM cross-encoder roughly 2-3x faster on CPU than
torch. The model is exported to ONNX once and cached in RERANK_ONNX_DIR
(default ``data/reranker_onnx``); later startups load the cached ONNX directly
so only the first-ever startup pays the export cost. A cross-process file lock
guards the export so the four gunicorn workers don't race to write it.

When ``optimum`` is not installed or the export/load fails, the torch
``CrossEncoder`` is used instead so startup never fails.

Both backends expose the same ``predict(pairs) -> list[float]`` interface, so
callers (app.main.rerank) are unaffected by which backend is active.
"""

import logging
import os

from app.config import config

logger = logging.getLogger("reranker")

# Flatten the model name into a safe local directory name (e.g. swap '/').
_ONNX_SUBDIR = "reranker_onnx"


class Reranker:
    """Cross-encoder reranker behind a single ``predict()`` interface."""

    def __init__(self, model_name: str, backend: str = "onnx"):
        self._onnx = None
        self._tokenizer = None
        self._torch = None
        if backend == "onnx":
            try:
                from optimum.onnxruntime import ORTModelForSequenceClassification
                from transformers import AutoTokenizer

                self._onnx, self._tokenizer = self._load_onnx(
                    model_name, ORTModelForSequenceClassification, AutoTokenizer
                )
                logger.info("reranker: using ONNX backend (%s)", model_name)
            except Exception as exc:
                logger.warning("reranker: ONNX backend unavailable (%s); falling back to torch", exc)
                self._onnx = None
        if self._onnx is None:
            from sentence_transformers import CrossEncoder

            self._torch = CrossEncoder(model_name, device="cpu")
            logger.info("reranker: using torch backend (%s)", model_name)

    def _load_onnx(self, model_name: str, orm_cls, tokenizer_cls):
        """Load the cached ONNX reranker, exporting it once on first use.

        Returns (model, tokenizer). Concurrent gunicorn workers are
        synchronized with an exclusive file lock around the export so exactly
        one worker writes the cache; the others wait and load it."""
        cache_dir = os.path.join(config.RERANK_ONNX_DIR, _ONNX_SUBDIR)
        ready = os.path.join(cache_dir, "model.onnx")
        if os.path.isfile(ready):
            return orm_cls.from_pretrained(cache_dir), tokenizer_cls.from_pretrained(cache_dir)

        os.makedirs(config.RERANK_ONNX_DIR, exist_ok=True)
        lock_path = os.path.join(config.RERANK_ONNX_DIR, ".reranker_onnx.lock")
        with open(lock_path, "w") as lock:
            import fcntl

            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                if os.path.isfile(ready):
                    return orm_cls.from_pretrained(cache_dir), tokenizer_cls.from_pretrained(cache_dir)
                model = orm_cls.from_pretrained(model_name, export=True)
                tokenizer = tokenizer_cls.from_pretrained(model_name)
                os.makedirs(cache_dir, exist_ok=True)
                model.save_pretrained(cache_dir)
                tokenizer.save_pretrained(cache_dir)
                return model, tokenizer
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Rerank relevance logits for (query, passage) pairs."""
        if self._onnx is not None:
            inputs = self._tokenizer(pairs, padding=True, truncation=True, return_tensors="pt")
            outputs = self._onnx(**inputs)
            logits = outputs.logits
            if logits.ndim == 2:
                return logits[:, 0].tolist()
            return logits.tolist()
        return self._torch.predict(pairs)