"""Stage 4 — embed chunks"""
from __future__ import annotations
from ..contracts import *  # noqa

import numpy as np


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
    except ImportError:
        raise ImportError("Please run: pip install sentence-transformers")

    model_name: str = cfg.get("embed", {}).get(
        "model", "paraphrase-multilingual-MiniLM-L12-v2"
    )

    model = SentenceTransformer(model_name)

    texts = []
    for chunk in chunks:
        # multilingual-e5 models require a "passage: " prefix for documents
        if "e5" in model_name.lower():
            texts.append(f"passage: {chunk.text}")
        else:
            texts.append(chunk.text)

    vectors = model.encode(
        texts,
        batch_size=cfg.get("embed", {}).get("batch_size", 32),
        show_progress_bar=True,
        normalize_embeddings=True,   # cosine sim = dot product after L2-norm
        convert_to_numpy=True,
    )

    return vectors  # shape: (n_chunks, dim)


def encode_query(query: str, cfg: dict) -> np.ndarray:
    """Embed a single query string. Used by the retriever at search time.

    Applies the same model and any required prefix as encode().

    Returns:
        np.ndarray of shape (embedding_dim,)
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError("Please run: pip install sentence-transformers")

    model_name: str = cfg.get("embed", {}).get(
        "model", "paraphrase-multilingual-MiniLM-L12-v2"
    )

    model = SentenceTransformer(model_name)

    text = f"query: {query}" if "e5" in model_name.lower() else query

    vector = model.encode(
        text,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )

    return vector  # shape: (dim,)
