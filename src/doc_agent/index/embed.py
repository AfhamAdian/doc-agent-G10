"""Stage 4 — embed chunks"""

from __future__ import annotations

import numpy as np

from ..config import resolve_device
from ..contracts import Chunk


def _batch_size(cfg: dict, device: str) -> int:
    embed_cfg = cfg.get("embed", {})
    key = "batch_size_gpu" if device == "cuda" else "batch_size_cpu"
    return int(embed_cfg.get(key, 16 if device == "cuda" else 4))


def encode(chunks: list[Chunk], cfg: dict) -> np.ndarray:
    """Embed with cfg['embed']['model'].

    Supported models (set via cfg['embed']['model']):
        - "paraphrase-multilingual-MiniLM-L12-v2"  (lightweight baseline)
        - "LaBSE"                                   (language-agnostic BERT, Google)
        - "intfloat/multilingual-e5-large"          (state-of-the-art multilingual)

    Returns:
        np.ndarray of shape (len(chunks), embedding_dim)
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError("Please run: pip install sentence-transformers") from exc

    model_name: str = cfg.get("embed", {}).get("model", "paraphrase-multilingual-MiniLM-L12-v2")
    expected_dim = int(cfg.get("embed", {}).get("dim", 0))
    if not chunks:
        return np.empty((0, expected_dim), dtype=np.float32)

    device = resolve_device(cfg)
    model = SentenceTransformer(model_name, device=device)

    texts = []
    for chunk in chunks:
        # multilingual-e5 models require a "passage: " prefix for documents
        if "e5" in model_name.lower():
            texts.append(f"passage: {chunk.text}")
        else:
            texts.append(chunk.text)

    vectors = model.encode(
        texts,
        batch_size=_batch_size(cfg, device),
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine sim = dot product after L2-norm
        convert_to_numpy=True,
    )

    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or len(vectors) != len(chunks):
        raise ValueError("Embedding model returned an invalid vector matrix")
    if expected_dim and vectors.shape[1] != expected_dim:
        raise ValueError(
            f"Embedding dimension mismatch: expected {expected_dim}, got {vectors.shape[1]}"
        )
    return vectors


def encode_query(query: str, cfg: dict) -> np.ndarray:
    """Embed a single query string. Used by the retriever at search time.

    Applies the same model and any required prefix as encode().

    Returns:
        np.ndarray of shape (embedding_dim,)
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError("Please run: pip install sentence-transformers") from exc

    model_name: str = cfg.get("embed", {}).get("model", "paraphrase-multilingual-MiniLM-L12-v2")

    device = resolve_device(cfg)
    model = SentenceTransformer(model_name, device=device)

    text = f"query: {query}" if "e5" in model_name.lower() else query

    vector = model.encode(
        text,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    vector = np.asarray(vector, dtype=np.float32)
    expected_dim = int(cfg.get("embed", {}).get("dim", 0))
    if expected_dim and vector.shape != (expected_dim,):
        raise ValueError(
            f"Query embedding dimension mismatch: expected {expected_dim}, got {vector.shape}"
        )
    return vector
