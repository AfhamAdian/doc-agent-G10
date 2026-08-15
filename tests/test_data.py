"""Embedding, vector-store, and corpus-gate tests."""

import json
import sys
import types

import numpy as np
import pytest

from doc_agent.contracts import Chunk, Page
from doc_agent.data.validate import validate, validate_pilot
from doc_agent.index.embed import encode
from doc_agent.index.store import build, load
from doc_agent.pipeline import _configure_pilot_output


def test_validate_rejects_small_corpus():
    with pytest.raises(ValueError, match="at least 300"):
        validate([])


def test_pilot_validation_allows_small_valid_corpus(tmp_path):
    image = tmp_path / "physics-001.png"
    image.write_bytes(b"rendered page")
    validate_pilot([Page(id=str(image), image_path=str(image), doc_id="physics")])


def test_pilot_uses_isolated_index_path():
    cfg = {
        "pilot": {"enabled": True, "index_path": "data/index-pilot"},
        "index": {"path": "data/index"},
    }
    assert _configure_pilot_output(cfg) is True
    assert cfg["index"]["path"] == "data/index-pilot"


def test_pilot_cannot_overwrite_production_index():
    cfg = {
        "pilot": {"enabled": True, "index_path": "data/index"},
        "index": {"path": "data/index"},
    }
    with pytest.raises(ValueError, match="must differ"):
        _configure_pilot_output(cfg)


def test_embedding_shape_normalization_and_faiss_round_trip(tmp_path, monkeypatch):
    class FakeSentenceTransformer:
        def __init__(self, model_name, device):
            assert model_name == "fake-e5"
            assert device == "cpu"

        def encode(self, texts, **kwargs):
            assert kwargs["batch_size"] == 4
            assert kwargs["normalize_embeddings"] is True
            return np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32)

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        types.SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    heldout = tmp_path / "heldout"
    heldout.mkdir()
    index_dir = tmp_path / "index"
    chunks = [
        Chunk(id="c1", doc_id="physics", text="one", page_ids=["physics-001.png"]),
        Chunk(id="c2", doc_id="biology", text="two", page_ids=["biology-001.png"]),
    ]
    cfg = {
        "device": "cpu",
        "ingest": {"heldout_dir": str(heldout)},
        "embed": {
            "model": "fake-e5",
            "dim": 3,
            "batch_size_cpu": 4,
            "batch_size_gpu": 16,
        },
        "index": {"type": "flat", "path": str(index_dir)},
        "_build_stats": {"page_count": 2, "ocr_word_count": 2},
    }
    vectors = encode(chunks, cfg)
    assert vectors.shape == (2, 3)
    assert np.allclose(np.linalg.norm(vectors, axis=1), 1.0)

    build(chunks, vectors, cfg)
    restored = load(cfg)
    assert len(restored) == 2
    assert [chunk.id for chunk in restored.search(vectors[0], k=2)][0] == "c1"
    metadata = json.loads((index_dir / "metadata.json").read_text(encoding="utf-8"))
    assert metadata["embedding_dimension"] == 3
    assert metadata["chunk_count"] == 2


def test_store_rejects_heldout_contamination(tmp_path):
    heldout = tmp_path / "heldout"
    heldout.mkdir()
    (heldout / "physics-017.png").write_bytes(b"held out")
    chunk = Chunk(
        id="c1",
        doc_id="physics",
        text="secret",
        page_ids=["data/interim/pages/physics/physics-017.png"],
    )
    cfg = {
        "device": "cpu",
        "ingest": {"heldout_dir": str(heldout)},
        "index": {"type": "flat", "path": str(tmp_path / "index")},
    }
    with pytest.raises(ValueError, match="Held-out pages"):
        build([chunk], np.array([[1.0, 0.0]], dtype=np.float32), cfg)
