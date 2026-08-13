"""Stage 3 — OCR/HTR (BASELINE = pretrained foundation, fine-tuned)"""
from __future__ import annotations
from ..contracts import *  # noqa

import uuid
import cv2
import numpy as np
from PIL import Image


class Reader:
    """Model set by cfg['ocr']. Baseline: pretrained TrOCR/Donut/Tesseract."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg["ocr"]
        self.model_name = self.cfg.get("model", "tesseract")
        self._model = None  # lazy-loaded on first call

    def _load_model(self):
        """Lazy-load the model on first use."""
        if self._model is not None:
            return

        if self.model_name == "tesseract":
            # No model object to load — pytesseract calls the CLI directly
            pass

        elif self.model_name == "easyocr":
            import easyocr
            # Bengali ('bn') and English ('en') together
            self._model = easyocr.Reader(["bn", "en"], gpu=self.cfg.get("gpu", False))

        elif self.model_name == "trocr":
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            checkpoint = self.cfg.get("checkpoint", "microsoft/trocr-base-printed")
            self._model = {
                "processor": TrOCRProcessor.from_pretrained(checkpoint),
                "model": VisionEncoderDecoderModel.from_pretrained(checkpoint),
            }

        elif self.model_name == "donut":
            from transformers import DonutProcessor, VisionEncoderDecoderModel
            checkpoint = self.cfg.get("checkpoint", "naver-clova-ix/donut-base")
            self._model = {
                "processor": DonutProcessor.from_pretrained(checkpoint, use_fast=False),
                "model": VisionEncoderDecoderModel.from_pretrained(checkpoint),
            }

        else:
            raise ValueError(f"Unknown OCR model: {self.model_name}")

    def transcribe_region(self, region: Region) -> str:
        """Read text from a single Region bounding box. Returns the transcribed string."""
        self._load_model()

        # Load the source image for this region
        img_bgr = cv2.imread(region.page_id)  # page_id is the image path in our pipeline
        if img_bgr is None:
            return ""

        x1, y1, x2, y2 = region.bbox
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
            results = self._model.readtext(crop_rgb, detail=0, paragraph=True)
            return "\n".join(results).strip()

        # --- TrOCR ---
        elif self.model_name == "trocr":
            import torch
            processor = self._model["processor"]
            model = self._model["model"]
            pixel_values = processor(images=pil_img, return_tensors="pt").pixel_values
            with torch.no_grad():
                generated_ids = model.generate(pixel_values)
            return processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()


        # --- Donut ---
        elif self.model_name == "donut":
            import torch
            import re
            processor = self._model["processor"]
            model = self._model["model"]
            task_prompt = "<s_cord-v2>"  # generic text reading prompt
            decoder_input_ids = processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
            pixel_values = processor(pil_img, return_tensors="pt").pixel_values
            with torch.no_grad():
                outputs = model.generate(pixel_values, decoder_input_ids=decoder_input_ids, max_length=512)
            sequence = processor.batch_decode(outputs.tolist())[0]
            # Strip special tokens
            sequence = re.sub(r"<.*?>", "", sequence).strip()
            return sequence

        return ""


def transcribe(regions: list[Region], cfg: dict) -> list[Chunk]:
    """Regions -> text chunks. Calls Reader for each non-figure region."""
    reader = Reader(cfg)
    chunks = []

    for region in regions:
        # Skip figure regions — we don't OCR diagrams
        if region.kind == "figure":
            continue

        text = reader.transcribe_region(region)
        if not text:
            continue

        chunk = Chunk(
            id=str(uuid.uuid4()),
            doc_id=region.page_id,
            text=text,
            page_ids=[region.page_id],
        )
        chunks.append(chunk)

    return chunks
