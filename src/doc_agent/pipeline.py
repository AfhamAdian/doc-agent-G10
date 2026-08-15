"""FIXED end-to-end order (Stages 0-9) + cross-cutting seams.
Do not reorder stages or remove hooks.run()/register_all() calls."""

from __future__ import annotations

from pathlib import Path

from . import config, hooks, wiring  # noqa: F401
from .agent import agent
from .contracts import Answer
from .data import validate
from .index import chunk, embed, store
from .ingest import enhance, loader, preprocess
from .retrieval import retriever
from .vision import layout, ocr


def _configure_pilot_output(cfg: dict) -> bool:
    """Select an isolated index path and return whether this is a pilot build."""
    pilot_cfg = cfg.get("pilot", {})
    pilot_enabled = bool(pilot_cfg.get("enabled", False))
    if not pilot_enabled:
        return False

    production_path = str(cfg.get("index", {}).get("path", "data/index"))
    pilot_path = str(pilot_cfg.get("index_path", "data/index-pilot"))
    if not pilot_path.strip():
        raise ValueError("pilot.index_path cannot be empty")

    if Path(pilot_path).resolve() == Path(production_path).resolve():
        raise ValueError("pilot.index_path must differ from the production index path")
    cfg.setdefault("index", {})["path"] = pilot_path
    return True


def build_knowledge_base(cfg: dict) -> None:
    pilot_enabled = _configure_pilot_output(cfg)
    wiring.register_all(cfg)  # wire cross-cutting features
    pages = loader.load_pages(cfg)
    pages = preprocess.run(pages, cfg)
    pages = enhance.run(pages, cfg)  # Stage 1 - enhancement (VAE/diffusion)
    if pilot_enabled:
        validate.validate_pilot(pages)
    else:
        validate.validate(pages)
    ingest_ctx = hooks.run(hooks.AFTER_INGEST, {"pages": pages})
    pages = ingest_ctx.get("pages", pages)
    regions = layout.detect(pages, cfg)  # Stage 2
    text = ocr.transcribe(regions, cfg)  # Stage 3
    ocr_ctx = hooks.run(hooks.AFTER_OCR, {"chunks": text})
    text = ocr_ctx.get("chunks", text)  # e.g. PII redaction on extracted text
    word_count = sum(len(item.text.split()) for item in text)
    if not pilot_enabled and word_count < 60_000:
        raise ValueError(f"Corpus has {word_count} usable OCR words; at least 60000 are required")
    chunks = chunk.split(text, cfg)  # Stage 4
    index_ctx = hooks.run(hooks.BEFORE_INDEX, {"chunks": chunks})
    chunks = index_ctx.get("chunks", chunks)
    if not chunks:
        raise ValueError("OCR and chunking produced no indexable text")
    vectors = embed.encode(chunks, cfg)
    cfg["_build_stats"] = {
        "pilot": pilot_enabled,
        "document_count": len({page.doc_id for page in pages}),
        "page_count": len(pages),
        "region_count": len(regions),
        "ocr_region_count": len(text),
        "ocr_word_count": word_count,
    }
    if pilot_enabled:
        cfg["_build_stats"].update(
            {
                "pilot_books": cfg.get("pilot", {}).get("books", ["physics"]),
                "pilot_max_pages_per_book": cfg.get("pilot", {}).get("max_pages_per_book", 20),
            }
        )
    store.build(chunks, vectors, cfg)


def answer(query_text: str, cfg: dict) -> Answer:
    wiring.register_all(cfg)
    r = retriever.Retriever(cfg)  # Stage 5
    return agent.Agent(cfg, r).run(query_text)  # Stage 6 (seams run inside the loop)
