"""Stage 3 — OCR/HTR (BASELINE = pretrained foundation, fine-tuned)"""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
from PIL import Image

from ..config import resolve_device
from ..contracts import Chunk, Region


class Reader:
    """Model set by cfg['ocr']. Baseline: pretrained TrOCR/Donut/Tesseract."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["ocr"]
        self.model_name = self.cfg.get("model", "tesseract")
        self.device = resolve_device(cfg)
        self._model: Any = None  # lazy-loaded on first call

    def _load_model(self) -> None:
        """Lazy-load the model on first use."""
        if self._model is not None:
            return

        if self.model_name == "tesseract":
            # No model object to load — pytesseract calls the CLI directly
            pass

        elif self.model_name == "easyocr":
            import easyocr

            languages = self.cfg.get("languages", ["bn", "en"])
            self._model = easyocr.Reader(languages, gpu=self.device == "cuda")

        elif self.model_name == "trocr":
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            checkpoint = self.cfg.get("checkpoint", "microsoft/trocr-base-printed")
            self._model = {
                "processor": TrOCRProcessor.from_pretrained(checkpoint),
                "model": VisionEncoderDecoderModel.from_pretrained(checkpoint).to(self.device),
            }

        elif self.model_name == "donut":
            from transformers import DonutProcessor, VisionEncoderDecoderModel

            checkpoint = self.cfg.get("checkpoint", "naver-clova-ix/donut-base")
            self._model = {
                "processor": DonutProcessor.from_pretrained(checkpoint, use_fast=False),
                "model": VisionEncoderDecoderModel.from_pretrained(checkpoint).to(self.device),
            }

        else:
            raise ValueError(f"Unknown OCR model: {self.model_name}")

    def transcribe_region(self, region: Region) -> str:
        """Read text from a single Region bounding box. Returns the transcribed string."""
        self._load_model()

        # Load the source image for this region
        img_bgr = cv2.imread(region.page_id)
        if img_bgr is None:
            raise ValueError(f"Could not read OCR source page: {region.page_id}")

        x1, y1, x2, y2 = region.bbox
        height, width = img_bgr.shape[:2]
        x1, x2 = max(0, x1), min(width, x2)
        y1, y2 = max(0, y1), min(height, y2)
        if x1 >= x2 or y1 >= y2:
            return ""
        crop_bgr = img_bgr[y1:y2, x1:x2]
        crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(crop_rgb)

        # --- Tesseract ---
        if self.model_name == "tesseract":
            import pytesseract

            lang = self.cfg.get("lang", "ben+eng")
            config = self.cfg.get("config", "--oem 3 --psm 6")
            return pytesseract.image_to_string(pil_img, lang=lang, config=config).strip()

        # --- EasyOCR ---
        elif self.model_name == "easyocr":
            results = self._model.readtext(
                crop_rgb,
                detail=0,
                paragraph=bool(self.cfg.get("paragraph", True)),
            )
            return "\n".join(results).strip()

        # --- TrOCR ---
        elif self.model_name == "trocr":
            import torch

            processor = self._model["processor"]
            model = self._model["model"]
            pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values.to(
                self.device
            )
            with torch.no_grad():
                generated_ids = model.generate(pixel_values)
            return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()

        # --- Donut ---
        elif self.model_name == "donut":
            import torch

            processor = self._model["processor"]
            model = self._model["model"]
            task_prompt = "<s_cord-v2>"  # generic text reading prompt
            decoder_input_ids = processor.tokenizer(
                task_prompt, add_special_tokens=False, return_tensors="pt"
            ).input_ids
            pixel_values = processor(pil_img, return_tensors="pt").pixel_values.to(self.device)
            with torch.no_grad():
                outputs = model.generate(
                    pixel_values, decoder_input_ids=decoder_input_ids, max_length=512
                )
            sequence = processor.batch_decode(outputs.tolist())[0]
            # Strip special tokens
            sequence = re.sub(r"<.*?>", "", sequence).strip()
            return sequence

        return ""


def _page_fingerprint(page_id: str, regions: list[Region], cfg: dict) -> str:
    digest = hashlib.sha256()
    with open(page_id, "rb") as image:
        for block in iter(lambda: image.read(1024 * 1024), b""):
            digest.update(block)
    settings = {
        "layout": cfg.get("layout", {}),
        "ocr": cfg.get("ocr", {}),
        "regions": [{"bbox": region.bbox, "kind": region.kind} for region in regions],
    }
    digest.update(json.dumps(settings, sort_keys=True, default=str).encode("utf-8"))
    return digest.hexdigest()


def _doc_id(page_id: str) -> str:
    stem = Path(page_id).stem
    prefix, separator, suffix = stem.rpartition("-")
    return prefix if separator and suffix.isdigit() else Path(page_id).parent.name


def _region_chunk(region: Region, text: str) -> Chunk:
    identity = f"{region.page_id}|{region.bbox}|{region.kind}|{text}"
    chunk_id = "ocr-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
    return Chunk(
        id=chunk_id,
        doc_id=_doc_id(region.page_id),
        text=text,
        page_ids=[region.page_id],
    )


def transcribe(regions: list[Region], cfg: dict) -> list[Chunk]:
    """Transcribe regions with deterministic, resumable per-page caching."""
    grouped: dict[str, list[Region]] = defaultdict(list)
    for region in regions:
        grouped[region.page_id].append(region)

    cache_dir = Path(cfg.get("ocr", {}).get("cache_dir", "data/interim/ocr"))
    cache_dir.mkdir(parents=True, exist_ok=True)
    reader: Reader | None = None
    chunks: list[Chunk] = []

    for page_id in sorted(grouped):
        page_regions = sorted(grouped[page_id], key=lambda item: (item.bbox[1], item.bbox[0]))
        fingerprint = _page_fingerprint(page_id, page_regions, cfg)
        cache_path = cache_dir / f"{Path(page_id).stem}-{fingerprint[:16]}.json"

        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            chunks.extend(Chunk.model_validate(item) for item in cached.get("chunks", []))
            continue

        if reader is None:
            reader = Reader(cfg)
        page_chunks: list[Chunk] = []
        for region in page_regions:
            if region.kind == "figure":
                continue
            text = reader.transcribe_region(region)
            if text:
                page_chunks.append(_region_chunk(region, text))

        payload = {
            "fingerprint": fingerprint,
            "page_id": page_id,
            "chunks": [chunk.model_dump(mode="json") for chunk in page_chunks],
        }
        cache_path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        chunks.extend(page_chunks)

    return chunks
