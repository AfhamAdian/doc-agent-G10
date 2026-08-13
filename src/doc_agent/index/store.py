"""Stage 4 — vector store"""
from __future__ import annotations
from ..contracts import *  # noqa

def build(chunks, vectors, cfg: dict) -> None:
    """Persist a vector index (cfg['index']['type']). IMPLEMENT."""
    """This will store the embeddings also load them too so implement it that wise"""
    raise NotImplementedError("Stage 4: build index")
def load(cfg: dict):
    raise NotImplementedError("Stage 4: load index")

