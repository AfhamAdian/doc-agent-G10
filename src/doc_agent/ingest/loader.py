"""Stage 1 — load scanned page-images"""

from __future__ import annotations

from pathlib import Path

from ..contracts import Page

_BOOKS = ("physics", "chemistry", "biology")


def _heldout_names(path: Path) -> set[str]:
    if not path.exists():
        return set()
    return {item.name for item in path.glob("*.png")}


def _render_pdf(
    pdf_path: Path,
    out_dir: Path,
    dpi: int,
    heldout: set[str],
    max_pages: int | None = None,
) -> list[Page]:
    try:
        import fitz
    except ImportError as exc:
        raise ImportError("PyMuPDF is required to render the corpus PDFs") from exc

    doc_id = pdf_path.stem
    pages: list[Page] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        document = fitz.open(pdf_path)
    except Exception as exc:
        raise ValueError(f"Unreadable PDF: {pdf_path}") from exc

    with document:
        if document.page_count == 0:
            raise ValueError(f"PDF contains no pages: {pdf_path}")

        page_count = document.page_count
        if max_pages is not None:
            page_count = min(page_count, max_pages)

        for page_index in range(page_count):
            filename = f"{doc_id}-{page_index + 1:03d}.png"
            if filename in heldout:
                continue

            image_path = out_dir / filename
            if not image_path.exists() or image_path.stat().st_size == 0:
                page = document.load_page(page_index)
                pixmap = page.get_pixmap(dpi=dpi, colorspace=fitz.csRGB, alpha=False)
                pixmap.save(image_path)

            relative_path = image_path.as_posix()
            pages.append(Page(id=relative_path, image_path=relative_path, doc_id=doc_id))

    return pages


def load_pages(cfg: dict) -> list[Page]:
    """Render configured corpus PDFs and return stable, non-held-out pages."""
    ingest_cfg = cfg.get("ingest", {})
    raw_dir = Path(ingest_cfg.get("raw_dir", "data/raw"))
    rendered_root = Path(ingest_cfg.get("rendered_dir", "data/interim/pages"))
    heldout_dir = Path(ingest_cfg.get("heldout_dir", "grading_kit/heldout_pages"))
    dpi = int(ingest_cfg.get("dpi", 300))
    if dpi <= 0:
        raise ValueError("ingest.dpi must be positive")

    pilot_cfg = cfg.get("pilot", {})
    pilot_enabled = bool(pilot_cfg.get("enabled", False))
    books = _BOOKS
    max_pages: int | None = None
    if pilot_enabled:
        configured_books = pilot_cfg.get("books", ["physics"])
        if not isinstance(configured_books, list) or not configured_books:
            raise ValueError("pilot.books must be a non-empty list")
        if any(not isinstance(book, str) or book not in _BOOKS for book in configured_books):
            raise ValueError(f"pilot.books may contain only: {', '.join(_BOOKS)}")
        if len(configured_books) != len(set(configured_books)):
            raise ValueError("pilot.books cannot contain duplicates")
        books = tuple(configured_books)
        max_pages = int(pilot_cfg.get("max_pages_per_book", 20))
        if max_pages <= 0:
            raise ValueError("pilot.max_pages_per_book must be positive")

    missing = [raw_dir / f"{book}.pdf" for book in books if not (raw_dir / f"{book}.pdf").is_file()]
    if missing:
        names = ", ".join(path.name for path in missing)
        raise FileNotFoundError(f"Missing corpus PDFs: {names}. Run scripts/get_data.sh first.")

    heldout = _heldout_names(heldout_dir) if ingest_cfg.get("exclude_heldout", True) else set()
    pages: list[Page] = []
    for book in books:
        pages.extend(
            _render_pdf(
                raw_dir / f"{book}.pdf",
                rendered_root / book,
                dpi,
                heldout,
                max_pages,
            )
        )
    return pages
