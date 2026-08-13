"""Stage 4 — chunk text"""
from __future__ import annotations
from ..contracts import *  # noqa

import re
import uuid


# ---------------------------------------------------------------------------
# Sentence splitting — robust against OCR inaccuracies
#
# OCR engines may produce:
#   - The correct Bangla danda     : ।  (U+0964)
#   - A pipe '|' instead of danda  : | (common OCR substitution — same visual shape)
#   - Newlines (Tesseract)         : \n
#   - Standard Latin punctuation   : . ! ?
# We handle all of these so that missing dandasalthough not ideal, do not cause
# the entire page to become one un-splittable blob.
# ---------------------------------------------------------------------------
_SENTENCE_BOUNDARY = re.compile(
    r'(?<=[।॥\|\.\!\?])\s+'   # after danda, double-danda, pipe, or Latin end marks
    r'|\n{2,}'                  # or after a blank line (paragraph break from Tesseract)
    r'|\n'                      # or after any newline
)

# Hard fallback: if no boundary found, split every FALLBACK_CHARS characters
# at the nearest word boundary so we don't cut mid-word.
FALLBACK_CHARS = 300


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences on semantic boundaries.

    Tries punctuation/newline splits first. If the result is a single huge
    block (i.e. no boundary was detected by the regex), falls back to
    splitting at word boundaries every FALLBACK_CHARS characters.
    """
    parts = _SENTENCE_BOUNDARY.split(text.strip())
    sentences = [s.strip() for s in parts if s.strip()]

    # If we got only one piece and it is very long, the OCR produced no boundaries.
    # Fall back to splitting on word boundaries every FALLBACK_CHARS characters.
    if len(sentences) == 1 and len(sentences[0]) > FALLBACK_CHARS:
        words = sentences[0].split()
        sentences = []
        current = []
        current_len = 0
        for word in words:
            current.append(word)
            current_len += len(word) + 1
            if current_len >= FALLBACK_CHARS:
                sentences.append(" ".join(current))
                current = []
                current_len = 0
        if current:
            sentences.append(" ".join(current))

    return sentences


# ---------------------------------------------------------------------------
# Semantic chunking: group sentences into size-bounded chunks with overlap
# ---------------------------------------------------------------------------
def split(chunks: list[Chunk], cfg: dict) -> list[Chunk]:
    """Re-chunk to cfg['index'] size/overlap using sentence-boundary semantic chunking (E4 NFR).

    Instead of cutting at a fixed character count, we split the text at natural
    sentence boundaries (Bangla danda and Latin punctuation), then greedily group
    sentences into chunks that fit within max_chars. Overlap is achieved by
    repeating the last overlap_chars worth of sentences at the start of the next chunk.

    cfg keys used:
        cfg['index']['chunk_size']    — max characters per chunk (default 500)
        cfg['index']['chunk_overlap'] — overlap characters between chunks (default 50)
    """
    index_cfg = cfg.get("index", {})
    max_chars: int = index_cfg.get("chunk_size", 500)
    overlap_chars: int = index_cfg.get("chunk_overlap", 50)

    result: list[Chunk] = []

    for chunk in chunks:
        sentences = _split_sentences(chunk.text)

        if not sentences:
            continue

        # Greedily group sentences into windows of at most max_chars
        current_sentences: list[str] = []
        current_len: int = 0

        for sentence in sentences:
            sentence_len = len(sentence)

            # If adding this sentence would overflow, flush and start a new chunk
            if current_sentences and current_len + sentence_len + 1 > max_chars:
                new_text = " ".join(current_sentences)
                result.append(Chunk(
                    id=str(uuid.uuid4()),
                    doc_id=chunk.doc_id,
                    text=new_text,
                    page_ids=chunk.page_ids,
                ))

                # Overlap: keep trailing sentences whose total length < overlap_chars
                overlap_sentences: list[str] = []
                overlap_len = 0
                for s in reversed(current_sentences):
                    if overlap_len + len(s) + 1 <= overlap_chars:
                        overlap_sentences.insert(0, s)
                        overlap_len += len(s) + 1
                    else:
                        break

                current_sentences = overlap_sentences
                current_len = overlap_len

            current_sentences.append(sentence)
            current_len += sentence_len + 1  # +1 for the space

        # Flush any remaining sentences as the final chunk
        if current_sentences:
            result.append(Chunk(
                id=str(uuid.uuid4()),
                doc_id=chunk.doc_id,
                text=" ".join(current_sentences),
                page_ids=chunk.page_ids,
            ))

    return result
