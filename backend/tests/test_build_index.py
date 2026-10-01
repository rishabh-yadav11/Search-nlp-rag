"""build_index.py: the collection is built, resumed, and payload-indexed.

Payload-index work matters most on a RESUMED build. create_payload_index used
to run only inside create_collection, which executes only when the collection is
(re)created -- so a collection predating a new field kept that field
unfilterable until a full destructive rebuild. These tests drive the real
main() with a fake Qdrant and fake embedding models to pin that the resumed path
indexes content_type and stores it on the points it upserts.

Import note: build_index imports sentence_transformers lazily inside main()
precisely so this module can be imported without the model stack.
"""
import json
import sys
import types

import build_index
import pytest
from qdrant_client.models import Modifier, PayloadSchemaType

from app.config import config


class _FakeDenseModel:
    def __init__(self, *a, **k):
        pass

    def encode(self, texts, **kwargs):
        import numpy as np

        return np.zeros((len(texts), config.EMBED_DIM), dtype="float32")


class _FakeSparseEmb:
    def __init__(self):
        import numpy as np

        self.indices = np.asarray([1], dtype=np.int32)
        self.values = np.asarray([1.0], dtype=np.float32)


class _FakeSparseModel:
    def __init__(self, *a, **k):
        pass

    def embed(self, texts):
        return iter([_FakeSparseEmb() for _ in texts])


@pytest.fixture
def fake_models(monkeypatch):
    """Stand in for torch-dependent embedders, installed as real modules."""
    st = types.ModuleType("sentence_transformers")
    st.SentenceTransformer = _FakeDenseModel
    fe = types.ModuleType("fastembed")
    fe.SparseTextEmbedding = _FakeSparseModel
    monkeypatch.setitem(sys.modules, "sentence_transformers", st)
    monkeypatch.setitem(sys.modules, "fastembed", fe)
    return st


class _FakeQdrant:
    """Records index creation and upserts; reports a compatible collection."""

    def __init__(self, *, exists: bool, compatible: bool = True):
        self.exists = exists
        self.compatible = compatible
        self.index_calls: list[tuple[str, str]] = []
        self.upserted: list = []
        self.created = False

    def get_collections(self):
        names = [config.QDRANT_COLLECTION] if self.exists else []
        return types.SimpleNamespace(collections=[types.SimpleNamespace(name=n) for n in names])

    def get_collection(self, *, collection_name):
        size = config.EMBED_DIM if self.compatible else config.EMBED_DIM + 1
        modifier = Modifier.IDF if self.compatible else None
        cfg = types.SimpleNamespace(
            params=types.SimpleNamespace(vectors={"dense": types.SimpleNamespace(size=size)}),
            sparse_vectors_config=types.SimpleNamespace(
                sparse={"modifier": modifier},
            ),
        )
        return types.SimpleNamespace(config=cfg, points_count=1)

    def create_collection(self, **kwargs):
        self.created = True

    def create_payload_index(self, collection_name, field, schema):
        self.index_calls.append((field, schema))

    def delete_collection(self, **kwargs):
        pass

    def upsert(self, *, collection_name, points, wait):
        self.upserted.extend(points)

    def close(self):
        pass


def _write_dataset(tmp_path, monkeypatch, rows):
    data = tmp_path / "articles.jsonl"
    data.write_text("".join(json.dumps(r) + "\n" for r in rows))
    monkeypatch.setattr(build_index, "DATA_PATH", str(data))
    monkeypatch.setattr(build_index, "CHECKPOINT_PATH", str(tmp_path / ".checkpoint"))
    return str(data)


_ROW = {
    "id": 1,
    "title": "Ola Electric IPO",
    "summary": "Funding news.",
    "body": "body text",
    "url": "https://www.vccircle.com/ola-electric-ipo",
    "published_date": "2025-06-01T00:00:00",
    "category": "Series A",
    "content_type": "Interview",
    "author_names": ["Alice"],
    "industry_names": ["Fintech"],
    "dealtype_names": ["Series A"],
    "tag_names": ["IPO", "VCC Startups"],
}


def test_create_collection_indexes_content_type(monkeypatch, tmp_path, fake_models):
    """A newly created collection indexes the field, so it is filterable."""
    client = _FakeQdrant(exists=False)
    monkeypatch.setattr(build_index, "QdrantClient", lambda *a, **k: client)
    _write_dataset(tmp_path, monkeypatch, [_ROW])

    build_index.main()

    assert client.created
    indexed = {f: s for f, s in client.index_calls}
    assert indexed["content_type"] == PayloadSchemaType.KEYWORD


def test_resumed_build_indexes_content_type_without_recreating(monkeypatch, tmp_path, fake_models):
    """A RESUMED build must still index the new field.

    ensure_collection returns False for an existing compatible collection, so
    when create_payload_index ran only there, content_type stayed unfilterable
    on exactly the collection a live deployment already has.
    """
    client = _FakeQdrant(exists=True, compatible=True)
    monkeypatch.setattr(build_index, "QdrantClient", lambda *a, **k: client)
    _write_dataset(tmp_path, monkeypatch, [_ROW])

    build_index.main()

    assert not client.created, "a compatible collection must be resumed, not recreated"
    indexed = {f: s for f, s in client.index_calls}
    assert indexed["content_type"] == PayloadSchemaType.KEYWORD

def test_resumed_build_stores_content_type_on_the_points_it_upserts(monkeypatch, tmp_path, fake_models):
    """The field reaches Qdrant, which is the whole point."""
    client = _FakeQdrant(exists=True, compatible=True)
    monkeypatch.setattr(build_index, "QdrantClient", lambda *a, **k: client)
    _write_dataset(tmp_path, monkeypatch, [_ROW])

    build_index.main()

    assert client.upserted, "the build must upsert something"
    assert [p.payload["content_type"] for p in client.upserted] == ["Interview"]


def test_resumed_build_indexes_and_stores_tag_names(monkeypatch, tmp_path, fake_models):
    """A tag filter needs both halves at once, on the resumed path.

    A resumed build is the state every deployed collection is in, and it is the
    path where the index call was historically skipped — so a stored-but-
    unindexed tag_names would make every /tag request match nothing.
    """
    client = _FakeQdrant(exists=True, compatible=True)
    monkeypatch.setattr(build_index, "QdrantClient", lambda *a, **k: client)
    _write_dataset(tmp_path, monkeypatch, [_ROW])

    build_index.main()

    assert {f: s for f, s in client.index_calls}["tag_names"] == PayloadSchemaType.KEYWORD
    assert [p.payload["tag_names"] for p in client.upserted] == [["IPO", "VCC Startups"]]


def test_incompatible_collection_is_recreated_and_indexed(monkeypatch, tmp_path, fake_models):
    """The destructive path still indexes the field (it goes via create_collection)."""
    client = _FakeQdrant(exists=True, compatible=False)
    monkeypatch.setattr(build_index, "QdrantClient", lambda *a, **k: client)
    _write_dataset(tmp_path, monkeypatch, [_ROW])

    build_index.main()

    assert client.created
    indexed = {f: s for f, s in client.index_calls}
    assert indexed["content_type"] == PayloadSchemaType.KEYWORD
