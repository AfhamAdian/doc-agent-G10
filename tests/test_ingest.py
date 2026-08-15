"""Ingestion and device-selection tests."""

from pathlib import Path

import fitz
import pytest

from doc_agent.config import resolve_device
from doc_agent.ingest.loader import load_pages
from doc_agent.ingest.preprocess import run


def _write_pdf(path: Path, text: str) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.save(path)
    document.close()


def test_loader_renders_stable_pages_excludes_heldout_and_reuses_cache(tmp_path):
    raw = tmp_path / "raw"
    rendered = tmp_path / "rendered"
    heldout = tmp_path / "heldout"
    raw.mkdir()
    heldout.mkdir()
    for book in ("physics", "chemistry", "biology"):
        _write_pdf(raw / f"{book}.pdf", f"{book} text")
    (heldout / "physics-001.png").write_bytes(b"held out marker")

    cfg = {
        "ingest": {
            "raw_dir": str(raw),
            "rendered_dir": str(rendered),
            "heldout_dir": str(heldout),
            "dpi": 72,
            "exclude_heldout": True,
        }
    }
    first = load_pages(cfg)
    assert [Path(page.image_path).name for page in first] == [
        "chemistry-001.png",
        "biology-001.png",
    ]
    mtimes = {page.image_path: Path(page.image_path).stat().st_mtime_ns for page in first}

    second = load_pages(cfg)
    assert [page.id for page in second] == [page.id for page in first]
    assert {page.image_path: Path(page.image_path).stat().st_mtime_ns for page in second} == mtimes


def test_clean_preprocess_is_pass_through(tmp_path):
    raw = tmp_path / "raw"
    rendered = tmp_path / "rendered"
    heldout = tmp_path / "heldout"
    raw.mkdir()
    heldout.mkdir()
    for book in ("physics", "chemistry", "biology"):
        _write_pdf(raw / f"{book}.pdf", book)
    cfg = {
        "ingest": {
            "raw_dir": str(raw),
            "rendered_dir": str(rendered),
            "heldout_dir": str(heldout),
            "dpi": 72,
        },
        "preprocess": {"deskew": False, "denoise": False, "binarize": False},
    }
    pages = load_pages(cfg)
    assert run(pages, cfg) == pages


def test_pilot_loads_only_selected_book_caps_source_pages_and_excludes_heldout(tmp_path):
    raw = tmp_path / "raw"
    rendered = tmp_path / "rendered"
    heldout = tmp_path / "heldout"
    raw.mkdir()
    heldout.mkdir()

    document = fitz.open()
    for page_number in range(1, 22):
        page = document.new_page()
        page.insert_text((72, 72), f"physics page {page_number}")
    document.save(raw / "physics.pdf")
    document.close()
    (heldout / "physics-017.png").write_bytes(b"held out marker")

    cfg = {
        "pilot": {
            "enabled": True,
            "books": ["physics"],
            "max_pages_per_book": 20,
        },
        "ingest": {
            "raw_dir": str(raw),
            "rendered_dir": str(rendered),
            "heldout_dir": str(heldout),
            "dpi": 72,
            "exclude_heldout": True,
        },
    }
    pages = load_pages(cfg)

    assert len(pages) == 19
    assert {page.doc_id for page in pages} == {"physics"}
    names = [Path(page.image_path).name for page in pages]
    assert "physics-017.png" not in names
    assert "physics-020.png" in names
    assert "physics-021.png" not in names


def test_pilot_rejects_unknown_book(tmp_path):
    cfg = {
        "pilot": {"enabled": True, "books": ["history"], "max_pages_per_book": 20},
        "ingest": {"raw_dir": str(tmp_path)},
    }
    with pytest.raises(ValueError, match="pilot.books may contain only"):
        load_pages(cfg)


def test_device_modes(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device({"device": "auto"}) == "cpu"
    assert resolve_device({"device": "cpu"}) == "cpu"
    with pytest.raises(RuntimeError, match="CUDA is not available"):
        resolve_device({"device": "cuda"})

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device({"device": "auto"}) == "cuda"
    assert resolve_device({"device": "cpu"}) == "cpu"
    assert resolve_device({"device": "cuda"}) == "cuda"
