"""Reranker tests: ONNX fast-path predict, torch fallback, and the ONNX
load/export/lock branches. optimum / sentence_transformers / transformers are
faked in sys.modules so nothing downloads or runs real inference; the ONNX cache
dir is redirected to a tmp path."""

import fcntl
import os
import sys
import types

import numpy as np

from app.config import config
from app.reranker import Reranker


class _RaisingModule:
    """A module-like object whose every attribute access raises (simulates an
    import failing, e.g. ``from optimum.onnxruntime import ...``)."""

    def __getattr__(self, name):
        raise AttributeError(name)


def _install(monkeypatch, name, module):
    monkeypatch.setitem(sys.modules, name, module)


def _make_onnx_fakes():
    """Fresh ORT + tokenizer fake classes plus a shared call recorder."""
    calls = {"from_pretrained": [], "save": []}

    class _Out:
        def __init__(self, logits):
            self.logits = logits

    class _FakeORT:
        def __init__(self, logits=None):
            self.logits = logits

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls["from_pretrained"].append((args, kwargs))
            return cls()

        def __call__(self, **inputs):
            return _Out(self.logits)

        def save_pretrained(self, path):
            calls["save"].append(("model", path))
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, "model.onnx"), "w") as f:
                f.write("x")

    class _FakeTokenizer:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            calls["from_pretrained"].append((args, kwargs))
            return cls()

        def __call__(self, pairs, **kwargs):
            return {"input_ids": pairs}

        def save_pretrained(self, path):
            calls["save"].append(("tokenizer", path))
            os.makedirs(path, exist_ok=True)

    return _FakeORT, _FakeTokenizer, calls


def _fake_onnx_modules(monkeypatch, orm_cls, tokenizer_cls):
    """Install working optimum + transformers fakes."""
    optimum = types.ModuleType("optimum")
    optimum_ort = types.ModuleType("optimum.onnxruntime")
    optimum_ort.ORTModelForSequenceClassification = orm_cls
    optimum.onnxruntime = optimum_ort
    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = tokenizer_cls
    _install(monkeypatch, "optimum", optimum)
    _install(monkeypatch, "optimum.onnxruntime", optimum_ort)
    _install(monkeypatch, "transformers", transformers)


def _fake_onnx_import_failure(monkeypatch):
    """optimum.onnxruntime import raises -> __init__ must fall back to torch."""
    _install(monkeypatch, "optimum", _RaisingModule())
    _install(monkeypatch, "optimum.onnxruntime", _RaisingModule())


def _fake_torch(monkeypatch, cross_encoder_cls):
    sentence_transformers = types.ModuleType("sentence_transformers")
    sentence_transformers.CrossEncoder = cross_encoder_cls
    _install(monkeypatch, "sentence_transformers", sentence_transformers)


def _fake_cross_encoder(predict_result=(0.9, 0.1)):
    class _FakeCrossEncoder:
        def __init__(self, model_name, device="cpu"):
            self.model_name = model_name
            self.device = device

        def predict(self, pairs):
            return list(predict_result)

    return _FakeCrossEncoder


def _record_flock(monkeypatch):
    """Patch fcntl.flock to record the ops; returns the recorder."""
    calls = []

    def fake_flock(fd, op):
        calls.append(op)

    monkeypatch.setattr(fcntl, "flock", fake_flock)
    return calls


def _cache_dir(tmp_path):
    return str(tmp_path / "reranker_onnx")


def _reranker_instance():
    return Reranker.__new__(Reranker)


def _onnx_model(logits):
    """A minimal ORT-like object with a fixed logits array."""
    outputs = type("Out", (), {"logits": logits})()
    return type("ONNX", (), {"__call__": lambda self, **kw: outputs})()


def _onnx_tokenizer():
    return type("Tok", (), {"__call__": lambda self, pairs, **kw: {"input_ids": pairs}})()


# --- predict branches ---


def test_onnx_predict_2d_logits_returns_first_column():
    rer = _reranker_instance()
    rer._onnx = _onnx_model(np.array([[0.5], [0.7]]))
    rer._tokenizer = _onnx_tokenizer()
    rer._torch = None

    assert rer.predict([("q", "a"), ("q", "b")]) == [0.5, 0.7]


def test_onnx_predict_flat_logits_returns_tolist():
    rer = _reranker_instance()
    rer._onnx = _onnx_model(np.array([0.5, 0.7]))
    rer._tokenizer = _onnx_tokenizer()
    rer._torch = None

    assert rer.predict([("q", "a"), ("q", "b")]) == [0.5, 0.7]


def test_torch_fallback_predict():
    rer = _reranker_instance()
    rer._onnx = None
    rer._tokenizer = None
    rer._torch = _fake_cross_encoder()("model-x")

    assert rer.predict([("q", "a")]) == [0.9, 0.1]


# --- __init__ failure/fallback branches ---


def test_init_onnx_import_failure_falls_back_to_torch(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RERANK_ONNX_DIR", str(tmp_path))
    _fake_onnx_import_failure(monkeypatch)
    _fake_torch(monkeypatch, _fake_cross_encoder())

    rer = Reranker("model-x")

    assert rer._onnx is None
    assert rer._torch is not None
    assert rer.predict([("q", "a")]) == [0.9, 0.1]


def test_init_onnx_export_failure_falls_back_to_torch(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RERANK_ONNX_DIR", str(tmp_path))
    orm_cls, tokenizer_cls, calls = _make_onnx_fakes()

    def boom_from_pretrained(*args, **kwargs):
        calls["from_pretrained"].append((args, kwargs))
        raise RuntimeError("export failed")

    orm_cls.from_pretrained = classmethod(boom_from_pretrained)
    _fake_onnx_modules(monkeypatch, orm_cls, tokenizer_cls)
    _fake_torch(monkeypatch, _fake_cross_encoder())

    rer = Reranker("model-x")

    assert rer._onnx is None
    assert rer._torch is not None
    assert rer.predict([("q", "a")]) == [0.9, 0.1]


def test_init_onnx_success_sets_onnx_backend(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RERANK_ONNX_DIR", str(tmp_path))
    orm_cls, tokenizer_cls, _calls = _make_onnx_fakes()
    _fake_onnx_modules(monkeypatch, orm_cls, tokenizer_cls)
    _fake_torch(monkeypatch, _fake_cross_encoder())

    rer = Reranker("model-x")

    assert rer._onnx is not None
    assert rer._tokenizer is not None
    assert rer._torch is None


# --- _load_onnx cache/export/lock branches ---


def test_load_onnx_cached_branch_skips_lock(monkeypatch, tmp_path):
    cache_dir = _cache_dir(tmp_path)
    os.makedirs(cache_dir, exist_ok=True)
    with open(os.path.join(cache_dir, "model.onnx"), "w") as f:
        f.write("x")
    monkeypatch.setattr(config, "RERANK_ONNX_DIR", str(tmp_path))
    flock_calls = _record_flock(monkeypatch)

    orm_cls, tokenizer_cls, calls = _make_onnx_fakes()
    rer = _reranker_instance()
    model, tokenizer = rer._load_onnx("model-x", orm_cls, tokenizer_cls)

    assert model is not None and tokenizer is not None
    assert calls["from_pretrained"] == [((cache_dir,), {}), ((cache_dir,), {})]
    assert calls["save"] == []
    assert flock_calls == []


def test_load_onnx_export_under_lock(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RERANK_ONNX_DIR", str(tmp_path))
    flock_calls = _record_flock(monkeypatch)
    cache_dir = _cache_dir(tmp_path)

    orm_cls, tokenizer_cls, calls = _make_onnx_fakes()
    rer = _reranker_instance()
    model, tokenizer = rer._load_onnx("model-x", orm_cls, tokenizer_cls)

    assert model is not None and tokenizer is not None
    assert calls["from_pretrained"] == [
        (("model-x",), {"export": True, "timeout": 300}),
        (("model-x",), {"timeout": 300}),
    ]
    assert calls["save"] == [("model", cache_dir), ("tokenizer", cache_dir)]
    assert os.path.isfile(os.path.join(cache_dir, "model.onnx"))
    assert flock_calls == [fcntl.LOCK_EX, fcntl.LOCK_UN]


def test_load_onnx_double_checked_lock_returns_cached(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RERANK_ONNX_DIR", str(tmp_path))
    cache_dir = _cache_dir(tmp_path)
    ready = os.path.join(cache_dir, "model.onnx")

    # While the second worker holds the lock, the first worker finishes the
    # export; the post-lock isfile() check then loads the cache instead.
    def flock_creates_model(fd, op):
        if op == fcntl.LOCK_EX:
            os.makedirs(cache_dir, exist_ok=True)
            with open(ready, "w") as f:
                f.write("x")

    monkeypatch.setattr(fcntl, "flock", flock_creates_model)

    orm_cls, tokenizer_cls, calls = _make_onnx_fakes()
    rer = _reranker_instance()
    model, tokenizer = rer._load_onnx("model-x", orm_cls, tokenizer_cls)

    assert model is not None and tokenizer is not None
    assert calls["from_pretrained"] == [((cache_dir,), {}), ((cache_dir,), {})]
    assert calls["save"] == []