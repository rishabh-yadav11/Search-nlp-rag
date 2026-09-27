"""Reranker tests: the torch-only construction and predict path.

The ONNX/optimum fast path and its load/export/lock tests were removed together
with the code they covered (optimum-onnx is not installable alongside the pinned
transformers 5.x, so that path could never run — see app/reranker.py).

``sentence_transformers`` is faked in ``sys.modules`` so nothing downloads a
model or runs real inference. A working ``optimum`` fake is installed in the
default-backend test to prove construction does not take an ONNX path even when
optimum *is* importable.
"""

import logging
import sys
import types

from app.reranker import Reranker


def _fake_torch(monkeypatch):
    """Install a fake sentence_transformers.CrossEncoder; returns (cls, calls)."""
    calls = []

    class _FakeCrossEncoder:
        def __init__(self, model_name, device="cpu"):
            calls.append({"model_name": model_name, "device": device})
            self.model_name = model_name
            self.device = device

        def predict(self, pairs):
            return [0.9 for _ in pairs]

    module = types.ModuleType("sentence_transformers")
    module.CrossEncoder = _FakeCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return _FakeCrossEncoder, calls


def _fake_onnx_importable(monkeypatch):
    """Install a *working* optimum.onnxruntime/transformers pair, recording any
    attempt to use the ONNX path, so a default construction can be shown not to
    take it. Returns the recorder."""
    calls = []

    class _RecordingORT:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append(("ort.from_pretrained", args, kwargs))
            return cls()

        def save_pretrained(self, path):
            calls.append(("ort.save_pretrained", (path,), {}))

        def __call__(self, **inputs):
            raise AssertionError("predict must not use the ONNX path")

    class _RecordingTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls.append(("tokenizer.from_pretrained", args, kwargs))
            return cls()

        def __call__(self, *args, **kwargs):
            raise AssertionError("predict must not use the ONNX path")

    optimum = types.ModuleType("optimum")
    optimum_ort = types.ModuleType("optimum.onnxruntime")
    optimum_ort.ORTModelForSequenceClassification = _RecordingORT
    optimum.onnxruntime = optimum_ort
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = _RecordingTokenizer
    monkeypatch.setitem(sys.modules, "optimum", optimum)
    monkeypatch.setitem(sys.modules, "optimum.onnxruntime", optimum_ort)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return calls


def test_default_backend_is_torch_even_when_optimum_is_importable(monkeypatch, caplog):
    onnx_calls = _fake_onnx_importable(monkeypatch)
    _cls, calls = _fake_torch(monkeypatch)

    with caplog.at_level(logging.DEBUG, logger="reranker"):
        rer = Reranker("model-x")

    # No ONNX export/load attempt of any kind, and no fallback warning: a
    # default construction goes straight to the torch CrossEncoder.
    assert onnx_calls == []
    assert not [rec for rec in caplog.records if "ONNX" in rec.getMessage()]
    assert rer.backend == "torch"
    assert calls == [{"model_name": "model-x", "device": "cpu"}]


def test_explicit_torch_backend_builds_cpu_cross_encoder(monkeypatch):
    _cls, calls = _fake_torch(monkeypatch)

    rer = Reranker("model-x", backend="torch")

    assert rer.backend == "torch"
    assert calls == [{"model_name": "model-x", "device": "cpu"}]


def test_unsupported_backend_warns_and_falls_back_to_torch(monkeypatch, caplog):
    _cls, calls = _fake_torch(monkeypatch)

    with caplog.at_level(logging.WARNING, logger="reranker"):
        rer = Reranker("model-x", backend="onnx")

    assert rer.backend == "torch"
    assert calls == [{"model_name": "model-x", "device": "cpu"}]
    assert any("onnx" in rec.getMessage() for rec in caplog.records)


def test_predict_delegates_pairs_to_torch_cross_encoder(monkeypatch):
    seen = []

    class _RecordingCrossEncoder:
        def __init__(self, model_name, device="cpu"):
            self.model_name = model_name

        def predict(self, pairs):
            seen.append(pairs)
            return [0.25, 0.75]

    module = types.ModuleType("sentence_transformers")
    module.CrossEncoder = _RecordingCrossEncoder
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    rer = Reranker("model-x")
    pairs = [("q", "a"), ("q", "b")]

    assert rer.predict(pairs) == [0.25, 0.75]
    assert seen == [pairs]
