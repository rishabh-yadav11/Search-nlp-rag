"""Cross-encoder reranker backed by a torch ``CrossEncoder``.

This module used to try an ONNX fast path (optimum/onnxruntime) first and fall
back to torch. That branch was removed: ``optimum-onnx`` pins
``transformers<4.58`` while this project pins ``transformers==5.10.1`` for the
CVE-2026-4372 / CVE-2026-5241 / CVE-2026-1839 fixes, so ``optimum`` is not
installable here (see backend/requirements.txt). With optimum absent the branch
could only ever raise a caught ``ImportError`` and log a fallback warning on
every gunicorn worker at startup, so it was dead code that cost startup time
and hid the real backend. The ONNX path may be reinstated if the transformers
pin is ever relaxed enough for ``optimum-onnx`` to install.

``backend`` is accepted so callers can keep passing ``config.RERANK_BACKEND``;
only ``"torch"`` is implemented. Any other value logs a warning and uses torch,
so a stale ``RERANK_BACKEND=onnx`` in a deployment's environment degrades the
same way the old fallback did instead of breaking startup.
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
