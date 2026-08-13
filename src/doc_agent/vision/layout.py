"""Stage 2 — layout detection / segmentation"""
from __future__ import annotations
from ..contracts import *  # noqa

import cv2


def detect(pages: list[Page], cfg: dict) -> list[Region]:
    """Detect text/table/figure/heading regions."""
    model_type = cfg.get("layout", {}).get("model", "doclayout_yolo")
    regions = []

    if model_type == "heuristic":
        # A naive heuristic: Treat the whole page as one giant text block.
        # This will fail spectacularly on multi-column tables and diagrams!
        for page in pages:
            img = cv2.imread(page.image_path)
            if img is None:
                continue
            h, w = img.shape[:2]
            regions.append(Region(
                page_id=page.id,
                bbox=(0, 0, w, h),
                kind="text"
            ))

    elif model_type == "doclayout_yolo":
        try:
            from ultralytics import YOLO
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise ImportError("Please run: pip install ultralytics huggingface_hub")

        # YOLO11 fine-tuned on DocLayNet (11 region types: Text, Title, Table, Figure, etc.)
        # Repo: Armaggheddon/yolo11-document-layout on HuggingFace
        checkpoint = cfg.get("layout", {}).get("checkpoint", None)
        if checkpoint is None:
            checkpoint = hf_hub_download(
                repo_id="Armaggheddon/yolo11-document-layout",
                filename="yolo11n_doc_layout.pt",
                repo_type="model"
            )
        model = YOLO(checkpoint)

        for page in pages:
            img = cv2.imread(page.image_path)
            if img is None:
                continue
            results = model(page.image_path)
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0])
                    raw_name = result.names[cls_id].lower()

                    # Map DocLayNet's 11 class names to our 4 Region kinds
                    if "table" in raw_name:
                        kind = "table"
                    elif "figure" in raw_name or "picture" in raw_name or "formula" in raw_name:
                        kind = "figure"
                    elif "title" in raw_name or "section" in raw_name or "heading" in raw_name:
                        kind = "heading"
                    else:
                        kind = "text"

                    regions.append(Region(
                        page_id=page.id,
                        bbox=(int(x1), int(y1), int(x2), int(y2)),
                        kind=kind
                    ))
    else:
        raise ValueError(f"Unknown layout model: {model_type}")

    return regions
