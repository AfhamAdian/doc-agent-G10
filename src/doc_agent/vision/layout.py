"""Stage 2 — layout detection / segmentation"""

from __future__ import annotations

import cv2

from ..config import resolve_device
from ..contracts import Page, Region


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
            regions.append(Region(page_id=page.id, bbox=(0, 0, w, h), kind="text"))

    elif model_type == "doclayout_yolo":
        try:
            from huggingface_hub import hf_hub_download
            from ultralytics import YOLO
        except ImportError as exc:
            raise ImportError("Please run: pip install ultralytics huggingface_hub") from exc

        # YOLO11 fine-tuned on DocLayNet (11 region types: Text, Title, Table, Figure, etc.)
        # Repo: Armaggheddon/yolo11-document-layout on HuggingFace
        layout_cfg = cfg.get("layout", {})
        checkpoint = layout_cfg.get("checkpoint")
        if checkpoint is None:
            checkpoint = hf_hub_download(
                repo_id=layout_cfg.get("repo_id", "Armaggheddon/yolo11-document-layout"),
                filename=layout_cfg.get("checkpoint_file", "yolo11n_doc_layout.pt"),
                repo_type="model",
            )
        model = YOLO(checkpoint)
        device = 0 if resolve_device(cfg) == "cuda" else "cpu"
        score_threshold = float(layout_cfg.get("score_thr", 0.5))

        for page in pages:
            img = cv2.imread(page.image_path)
            if img is None:
                continue
            results = model(
                page.image_path,
                conf=score_threshold,
                device=device,
                verbose=False,
            )
            for result in results:
                for box in result.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    cls_id = int(box.cls[0])
                    raw_name = result.names[cls_id].lower()

                    # Map DocLayNet's 11 class names to our 4 Region kinds
                    if "table" in raw_name:
                        kind = "table"
                    elif "formula" in raw_name:
                        kind = "text"
                    elif "figure" in raw_name or "picture" in raw_name:
                        kind = "figure"
                    elif "title" in raw_name or "section" in raw_name or "heading" in raw_name:
                        kind = "heading"
                    else:
                        kind = "text"

                    regions.append(
                        Region(
                            page_id=page.id, bbox=(int(x1), int(y1), int(x2), int(y2)), kind=kind
                        )
                    )
    else:
        raise ValueError(f"Unknown layout model: {model_type}")

    return sorted(regions, key=lambda region: (region.page_id, region.bbox[1], region.bbox[0]))
