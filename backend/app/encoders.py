"""Query-time dense encoder behind a single ``encode()`` interface.

The index is built with a full-precision sentence-transformers bge model, but
at query time fastembed's ONNX (INT8) variant of the same model runs roughly
3-5x faster on CPU while producing L2-normalized vectors with the same
direction, so cosine search against the existing index stays correct.

If fastembed cannot load the model (e.g. model not downloaded yet or no
network), we fall back to the torch sentence-transformers model so startup
never fails.
"""

import logging

logger = logging.getLogger("encoders")


class DenseEncoder:
    """Dense embedder with a fastembed/ONNX fast path and a torch fallback."""

    def __init__(self, model_name: str, device: str, threads: int):
        self._fallback = None
        self._model = None
        try:
            from fastembed import TextEmbedding

            cuda = device.lower() not in ("cpu", "auto")
            self._model = TextEmbedding(model_name=model_name, cuda=cuda, threads=threads)
        except Exception as exc:
            logger.warning("fastembed dense encoder unavailable (%s); falling back to torch", exc)
            self._model = None
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._fallback = SentenceTransformer(model_name, device=device)

    def encode(self, text: str):
        """Embed one query string into an L2-normalized vector."""
        if self._fallback is not None:
            return self._fallback.encode(text, normalize_embeddings=True)
        return next(iter(self._model.embed([text], batch_size=1)))