"""Stage 1 — deskew / denoise / binarize / augment"""

from __future__ import annotations

from pathlib import Path

from ..contracts import Page


def run(pages: list[Page], cfg: dict) -> list[Page]:
    """Validate clean rendered pages and optionally apply classical cleanup.

    The committed configuration is deliberately pass-through: these PDFs are
    vector-rendered, with no physical skew or scan noise to repair.
    """
    options = cfg.get("preprocess", {})
    enabled = any(bool(options.get(name, False)) for name in ("deskew", "denoise", "binarize"))

    try:
        import cv2
    except ImportError as exc:
        raise ImportError("opencv-python-headless is required for preprocessing") from exc

    processed: list[Page] = []
    for page in pages:
        source = Path(page.image_path)
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise ValueError(f"Unreadable rendered page: {source}")

        if not enabled:
            processed.append(page)
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if options.get("denoise", False):
            gray = cv2.medianBlur(gray, 3)
        if options.get("binarize", False):
            _, gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # Vector-rendered pages are structurally unskewed. The deskew flag is
        # accepted for configuration compatibility but intentionally performs
        # no rotation unless a future corpus provides a measured angle.

        out_dir = Path(cfg.get("ingest", {}).get("rendered_dir", "data/interim/pages"))
        out_path = out_dir / "preprocessed" / page.doc_id / source.name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(out_path), gray):
            raise OSError(f"Could not write preprocessed page: {out_path}")
        relative = out_path.as_posix()
        processed.append(Page(id=relative, image_path=relative, doc_id=page.doc_id))

    return processed
