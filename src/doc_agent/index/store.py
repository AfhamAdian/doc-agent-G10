"""Stage 4 — vector store"""

from __future__ import annotations

import hashlib
import json
import pathlib
from typing import Any

import numpy as np

from ..config import resolve_device
from ..contracts import Chunk
from ..logging_conf import get_logger

LOGGER = get_logger(__name__)

# ---------------------------------------------------------------------------
# VectorStore — returned by load(), used by retriever.retrieve()
# ---------------------------------------------------------------------------


class VectorStore:
    """Thin wrapper around a FAISS index + the original Chunk objects.

    Usage:
        store = load(cfg)
        chunks = store.search(query_vector, k=5)  # returns list[Chunk] with .score set
    """

    def __init__(self, index: Any, chunks: list[Chunk]) -> None:
        self._index = index
        self._chunks = chunks  # same order as vectors in the index

    def search(self, query_vec: np.ndarray, k: int = 5) -> list[Chunk]:
        """Return top-k Chunks ranked by cosine similarity.

        Assumes query_vec is already L2-normalised (matches how embed.encode_query works).
        Sets chunk.score to the cosine similarity for use by retriever.is_weak().
        """
        q = np.atleast_2d(query_vec).astype("float32")
        k = min(k, len(self._chunks))
        scores, indices = self._index.search(q, k)

        results: list[Chunk] = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx < 0:  # FAISS returns -1 for empty slots
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
    except ImportError as exc:
        raise ImportError("Please run: pip install faiss-cpu") from exc

    index_cfg = cfg.get("index", {})
    out_dir = pathlib.Path(index_cfg.get("path", "data/index"))
    out_dir.mkdir(parents=True, exist_ok=True)

    vecs = np.array(vectors, dtype="float32")
    if not chunks or vecs.ndim != 2 or vecs.shape[0] != len(chunks):
        raise ValueError("Chunk and embedding counts must match and be non-empty")
    norms = np.linalg.norm(vecs, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-3):
        raise ValueError("Embeddings must be L2-normalized before indexing")

    heldout_dir = pathlib.Path(
        cfg.get("ingest", {}).get("heldout_dir", "grading_kit/heldout_pages")
    )
    heldout_names = (
        {path.name for path in heldout_dir.glob("*.png")} if heldout_dir.exists() else set()
    )
    contaminated = sorted(
        {
            pathlib.Path(page_id).name
            for chunk in chunks
            for page_id in chunk.page_ids
            if pathlib.Path(page_id).name in heldout_names
        }
    )
    if contaminated:
        raise ValueError(f"Held-out pages cannot be indexed: {', '.join(contaminated)}")

    dim = vecs.shape[1]
    index_type = index_cfg.get("type", "flat")

    if index_type == "hnsw":
        m = index_cfg.get("hnsw_m", 32)  # number of neighbours per node
        index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = 200  # build-time accuracy/speed trade-off
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

    public_cfg = {key: value for key, value in cfg.items() if not str(key).startswith("_")}
    config_hash = hashlib.sha256(
        json.dumps(public_cfg, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()
    metadata = {
        **cfg.get("_build_stats", {}),
        "chunk_count": len(chunks),
        "embedding_dimension": dim,
        "embedding_model": cfg.get("embed", {}).get("model"),
        "index_type": index_type,
        "index_bytes": (out_dir / "index.faiss").stat().st_size,
        "device": resolve_device(cfg),
        "config_sha256": config_hash,
    }
    (out_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    LOGGER.info(
        "built index type=%s chunks=%d dim=%d path=%s",
        index_type,
        len(chunks),
        dim,
        out_dir,
    )


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
    except ImportError as exc:
        raise ImportError("Please run: pip install faiss-cpu") from exc

    index_cfg = cfg.get("index", {})
    out_dir = pathlib.Path(index_cfg.get("path", "data/index"))

    index_path = out_dir / "index.faiss"
    chunks_path = out_dir / "chunks.jsonl"

    if not index_path.exists() or not chunks_path.exists():
        raise FileNotFoundError(f"No index found at {out_dir}. Run build_index.sh first.")

    index = faiss.read_index(str(index_path))

    chunks: list[Chunk] = []
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(Chunk.model_validate_json(line))

    if index.ntotal != len(chunks):
        raise ValueError(
            f"Index/chunk mismatch: index has {index.ntotal}, metadata has {len(chunks)}"
        )
    LOGGER.info("loaded index chunks=%d path=%s", len(chunks), out_dir)
    return VectorStore(index, chunks)
