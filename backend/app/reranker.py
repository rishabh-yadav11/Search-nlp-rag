"""Cross-encoder reranker backed by a torch ``CrossEncoder``.

There is deliberately no ONNX fast path: ``optimum-onnx`` pins ``transformers<4.58`` while this
project pins a much newer ``transformers`` for CVE fixes, so ``optimum`` is not installable here
(see backend/requirements.txt). With optimum absent such a branch could only raise a caught
``ImportError`` and log a fallback warning on every worker at startup. It may be reinstated if
the pin is ever relaxed enough for ``optimum-onnx`` to install.

``backend`` is accepted so callers can keep passing ``config.RERANK_BACKEND``; only ``"torch"``
is implemented, and any other value warns and uses torch so a stale ``RERANK_BACKEND=onnx``
degrades instead of breaking startup.
"""

import logging

logger = logging.getLogger("reranker")


class Reranker:
    """Cross-encoder reranker behind a single ``predict()`` interface."""

    def __init__(self, model_name: str, backend: str = "torch"):
        if backend != "torch":
            logger.warning(
                "reranker: backend %r is not available (only 'torch' is supported); using torch",
                backend,
            )
        from sentence_transformers import CrossEncoder

        self.backend = "torch"
        self._torch = CrossEncoder(model_name, device="cpu")
        logger.info("reranker: using torch backend (%s)", model_name)

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        """Rerank relevance logits for (query, passage) pairs."""
        return self._torch.predict(pairs)
