"""Stage 4 — vector store"""
from __future__ import annotations
from ..contracts import *  # noqa

import json
import pathlib
import numpy as np


# ---------------------------------------------------------------------------
# VectorStore — returned by load(), used by retriever.retrieve()
# ---------------------------------------------------------------------------

class VectorStore:
    """Thin wrapper around a FAISS index + the original Chunk objects.

    Usage:
        store = load(cfg)
        chunks = store.search(query_vector, k=5)  # returns list[Chunk] with .score set
    """

    def __init__(self, index, chunks: list[Chunk]) -> None:
        self._index = index
        self._chunks = chunks   # same order as vectors in the index

    def search(self, query_vec: np.ndarray, k: int = 5) -> list[Chunk]:
        """Return top-k Chunks ranked by cosine similarity.

        Assumes query_vec is already L2-normalised (matches how embed.encode_query works).
        Sets chunk.score to the cosine similarity for use by retriever.is_weak().
        """
        q = np.atleast_2d(query_vec).astype("float32")
        k = min(k, len(self._chunks))
        scores, indices = self._index.search(q, k)

        results: list[Chunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:          # FAISS returns -1 for empty slots
                continue
            chunk = self._chunks[idx].model_copy()
            chunk.score = float(score)
            results.append(chunk)
        return results

    def __len__(self) -> int:
        return len(self._chunks)


# ---------------------------------------------------------------------------
# build — called once by scripts/build_index.sh
# ---------------------------------------------------------------------------

def build(chunks: list[Chunk], vectors: np.ndarray, cfg: dict) -> None:
    """Persist a vector index to disk.

    cfg keys used:
        cfg['index']['path']  — directory to write to (default: 'data/index')
        cfg['index']['type']  — 'flat' (exact) or 'hnsw' (approximate, default: 'flat')
        cfg['index']['hnsw_m'] — HNSW M parameter (default: 32, only used when type='hnsw')
    """
    try:
        import faiss
    except ImportError:
        raise ImportError("Please run: pip install faiss-cpu")

    index_cfg = cfg.get("index", {})
    out_dir = pathlib.Path(index_cfg.get("path", "data/index"))
    out_dir.mkdir(parents=True, exist_ok=True)

    vecs = np.array(vectors, dtype="float32")
    dim = vecs.shape[1]
    index_type = index_cfg.get("type", "flat")

    if index_type == "hnsw":
        m = index_cfg.get("hnsw_m", 32)        # number of neighbours per node
        index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200         # build-time accuracy/speed trade-off
    else:
        # Flat = exact brute-force cosine search (vectors are L2-normalised)
        index = faiss.IndexFlatIP(dim)

    index.add(vecs)

    # Persist FAISS index
    faiss.write_index(index, str(out_dir / "index.faiss"))

    # Persist chunk metadata alongside the index
    with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(chunk.model_dump_json() + "\n")

    print(f"[store] Built {index_type} index: {len(chunks)} chunks, dim={dim} → {out_dir}")


# ---------------------------------------------------------------------------
# load — called by Retriever.__init__() at query time
# ---------------------------------------------------------------------------

def load(cfg: dict) -> VectorStore:
    """Load a persisted index from disk and return a searchable VectorStore.

    cfg keys used:
        cfg['index']['path'] — directory to read from (default: 'data/index')
    """
    try:
        import faiss
    except ImportError:
        raise ImportError("Please run: pip install faiss-cpu")

    index_cfg = cfg.get("index", {})
    out_dir = pathlib.Path(index_cfg.get("path", "data/index"))

    index_path = out_dir / "index.faiss"
    chunks_path = out_dir / "chunks.jsonl"

    if not index_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(
            f"No index found at {out_dir}. Run build_index.sh first."
        )

    index = faiss.read_index(str(index_path))

    chunks: list[Chunk] = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(Chunk.model_validate_json(line))

    print(f"[store] Loaded index: {len(chunks)} chunks from {out_dir}")
    return VectorStore(index, chunks)
