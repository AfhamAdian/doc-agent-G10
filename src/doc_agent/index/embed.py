"""Stage 4 — embed chunks"""
from __future__ import annotations
from ..contracts import *  # noqa

def encode(chunks: list[Chunk], cfg: dict):
    """Embed with cfg['embed']['model']. IMPLEMENT."""
    "I will implement it here"
    """2. src/doc_agent/index/embed.py → encode(chunks: list[Chunk], cfg: dict) -> np.ndarray
    - Input: the split chunks + cfg['embed'] = {model: all-MiniLM-L6-v2, dim: 384}.
    - Output: np.float32 array shape (N, 384), row i = embedding of chunks[i] (order must match). Normalize rows (L2) so downstream cosine scores are meaningful.
    - Who uses it: only pipeline.py, feeding store.build(chunks, vectors, cfg). First call downloads the model (network needed)."""
    raise NotImplementedError("Stage 4: embed")

