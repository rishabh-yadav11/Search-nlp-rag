

import logging

logger = logging.getLogger("reranker")


class Reranker:
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
        return self._torch.predict(pairs)
