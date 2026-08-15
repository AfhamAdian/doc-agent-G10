"""Data — data schema/quality validation at ingest"""

from __future__ import annotations

from pathlib import Path

from ..contracts import Page


def _validate_page_contracts(pages: list[Page]) -> None:
    if not pages:
        raise ValueError("Corpus has no usable pages")

    ids = [page.id for page in pages]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate page IDs found in the corpus")

    missing = [page.image_path for page in pages if not Path(page.image_path).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing rendered page: {missing[0]}")


def validate(pages: list[Page]) -> None:
    """Enforce the page-level corpus contract before expensive inference."""
    if len(pages) < 300:
        raise ValueError(f"Corpus has {len(pages)} usable pages; at least 300 are required")

    _validate_page_contracts(pages)

    if len({page.doc_id for page in pages}) < 2:
        raise ValueError("Document-level separation requires more than one source document")


def validate_pilot(pages: list[Page]) -> None:
    """Validate shared page contracts without applying full-corpus size gates."""
    _validate_page_contracts(pages)
