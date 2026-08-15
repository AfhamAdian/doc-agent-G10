"""OCR cache and deterministic chunking tests."""

from PIL import Image

from doc_agent.contracts import Chunk, Region
from doc_agent.index.chunk import split
from doc_agent.vision.ocr import Reader, transcribe


def test_ocr_cache_and_ids_are_deterministic(tmp_path, monkeypatch):
    image_path = tmp_path / "physics-001.png"
    Image.new("RGB", (100, 50), "white").save(image_path)
    region = Region(page_id=str(image_path), bbox=(0, 0, 100, 50), kind="text")
    calls = []

    def fake_transcribe(self, item):
        calls.append(item.page_id)
        return "বল F = ma।"

    monkeypatch.setattr(Reader, "transcribe_region", fake_transcribe)
    cfg = {
        "device": "cpu",
        "layout": {"model": "test"},
        "ocr": {"model": "easyocr", "cache_dir": str(tmp_path / "cache")},
    }
    first = transcribe([region], cfg)
    second = transcribe([region], cfg)
    assert first == second
    assert len(first) == 1
    assert first[0].id.startswith("ocr-")
    assert calls == [str(image_path)]


def test_semantic_chunk_ids_are_reproducible():
    source = Chunk(
        id="ocr-source",
        doc_id="physics",
        text=("প্রথম বাক্য। " * 80) + "F = ma।",
        page_ids=["physics-001.png"],
    )
    cfg = {"index": {"chunk_size": 500, "chunk_overlap": 50}}
    first = split([source], cfg)
    second = split([source], cfg)
    assert first == second
    assert len({chunk.id for chunk in first}) == len(first)
    assert all(chunk.page_ids == source.page_ids for chunk in first)
