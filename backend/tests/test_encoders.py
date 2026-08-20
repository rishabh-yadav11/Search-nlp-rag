"""DenseEncoder tests: fastembed/ONNX fast-path init, torch fallback on load
failure, and both encode branches. fastembed / sentence_transformers are faked
in sys.modules so nothing downloads a model or runs real inference."""

import sys
import types

import numpy as np

from app.encoders import DenseEncoder


def _install(monkeypatch, name, module):
    monkeypatch.setitem(sys.modules, name, module)


def _install_fastembed(monkeypatch, model_cls):
    fastembed = types.ModuleType("fastembed")
    fastembed.TextEmbedding = model_cls
    _install(monkeypatch, "fastembed", fastembed)


def _install_sentence_transformers(monkeypatch, st_cls):
    sentence_transformers = types.ModuleType("sentence_transformers")
    sentence_transformers.SentenceTransformer = st_cls
    _install(monkeypatch, "sentence_transformers", sentence_transformers)


def _make_text_embedding(vectors=None, raise_on_init=False):
    calls = {"init": [], "embed": []}

    class _FakeTextEmbedding:
        def __init__(self, model_name, cuda, threads):
            calls["init"].append((model_name, cuda, threads))
            if raise_on_init:
                raise RuntimeError("model unavailable")

        def embed(self, texts, batch_size=1):
            calls["embed"].append((texts, batch_size))
            yield from vectors or []

    return _FakeTextEmbedding, calls


def _make_sentence_transformer(result=None):
    calls = {"init": [], "encode": []}

    class _FakeSentenceTransformer:
        def __init__(self, model_name, device):
            calls["init"].append((model_name, device))

        def encode(self, text, normalize_embeddings=True):
            calls["encode"].append((text, normalize_embeddings))
            return result

    return _FakeSentenceTransformer, calls


# --- __init__ fastembed/ONNX success ---


def test_init_fastembed_success_sets_onnx_model(monkeypatch):
    model_cls, fe_calls = _make_text_embedding()
    _install_fastembed(monkeypatch, model_cls)

    enc = DenseEncoder("model-x", "cpu", 2)

    assert enc._model is not None
    assert enc._fallback is None
    assert fe_calls["init"] == [("model-x", False, 2)]


def test_init_fastembed_success_cuda_device(monkeypatch):
    model_cls, fe_calls = _make_text_embedding()
    _install_fastembed(monkeypatch, model_cls)

    enc = DenseEncoder("model-x", "cuda", 2)

    assert enc._model is not None
    assert enc._fallback is None
    assert fe_calls["init"] == [("model-x", True, 2)]


# --- __init__ fastembed load failure -> torch fallback ---


def test_init_fastembed_failure_falls_back_to_torch(monkeypatch):
    model_cls, fe_calls = _make_text_embedding(raise_on_init=True)
    _install_fastembed(monkeypatch, model_cls)
    st_cls, st_calls = _make_sentence_transformer()
    _install_sentence_transformers(monkeypatch, st_cls)

    enc = DenseEncoder("model-x", "cpu", 2)

    assert enc._model is None
    assert enc._fallback is not None
    assert fe_calls["init"] == [("model-x", False, 2)]
    assert st_calls["init"] == [("model-x", "cpu")]


# --- encode branches ---


def test_encode_fastembed_generator_path():
    model_cls, fe_calls = _make_text_embedding(vectors=[np.array([0.1, 0.2])])
    enc = DenseEncoder.__new__(DenseEncoder)
    enc._model = model_cls("model-x", False, 2)
    enc._fallback = None

    result = enc.encode("query")

    assert np.allclose(result, [0.1, 0.2])
    assert fe_calls["embed"] == [(["query"], 1)]


def test_encode_torch_fallback():
    st_cls, st_calls = _make_sentence_transformer(result=[0.3, 0.4])
    enc = DenseEncoder.__new__(DenseEncoder)
    enc._model = None
    enc._fallback = st_cls("model-x", "cpu")

    result = enc.encode("query")

    assert result == [0.3, 0.4]
    assert st_calls["encode"] == [("query", True)]