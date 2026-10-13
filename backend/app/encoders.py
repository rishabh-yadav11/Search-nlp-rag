

import logging
from typing import Any

logger = logging.getLogger("encoders")


class DenseEncoder:
    def __init__(self, model_name: str, device: str, threads: int):
        self._fallback: Any = None
        self._model: Any = None
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

    def encode(self, text: str) -> Any:

        if self._fallback is not None:
            return self._fallback.encode(text, normalize_embeddings=True)
        vec = next(iter(self._model.embed([text], batch_size=1)))
        norm = (vec ** 2).sum() ** 0.5
        if norm > 0:
            vec = vec / norm
        return vec