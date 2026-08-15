#!/usr/bin/env bash
# A2 — reproducibly build and validate the vector index from the raw corpus.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

if [[ -n "${PYTHON:-}" ]]; then
    PYTHON_BIN="$PYTHON"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
    PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
    PYTHON_BIN="$(command -v python)"
else
    echo "No Python interpreter found. Set PYTHON or create .venv first." >&2
    exit 1
fi

echo "Using Python: $PYTHON_BIN"

# Fail before expensive rendering/OCR if the environment or configuration is invalid.
"$PYTHON_BIN" - <<'PY'
from doc_agent import config

cfg = config.load()
for package in ("faiss", "sentence_transformers"):
    try:
        __import__(package)
    except ImportError as exc:
        raise SystemExit(
            f"Missing required package {package!r}. Run `make setup` first."
        ) from exc

if not cfg.get("embed", {}).get("model"):
    raise SystemExit("configs/config.yaml must define embed.model")
if not cfg.get("index", {}).get("path"):
    raise SystemExit("configs/config.yaml must define index.path")
PY

echo "Building knowledge-base index from the configured raw corpus..."
"$PYTHON_BIN" scripts/run_index.py

# Load the artifact that was just built and verify its persisted contract.
"$PYTHON_BIN" - <<'PY'
from __future__ import annotations

import json
from pathlib import Path

from doc_agent import config
from doc_agent.index import store

cfg = config.load()
pilot_cfg = cfg.get("pilot", {})
pilot_enabled = bool(pilot_cfg.get("enabled", False))

if pilot_enabled:
    output_path = str(pilot_cfg.get("index_path", "data/index-pilot"))
    cfg.setdefault("index", {})["path"] = output_path
else:
    output_path = str(cfg.get("index", {}).get("path", "data/index"))

index_dir = Path(output_path)
metadata_path = index_dir / "metadata.json"
if not metadata_path.is_file():
    raise SystemExit(f"Build did not produce metadata: {metadata_path}")

metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
vector_store = store.load(cfg)

expected_model = cfg.get("embed", {}).get("model")
expected_dimension = int(cfg.get("embed", {}).get("dim", 0))
if metadata.get("embedding_model") != expected_model:
    raise SystemExit(
        "Embedding model mismatch: "
        f"expected {expected_model!r}, got {metadata.get('embedding_model')!r}"
    )
if expected_dimension and metadata.get("embedding_dimension") != expected_dimension:
    raise SystemExit(
        "Embedding dimension mismatch: "
        f"expected {expected_dimension}, got {metadata.get('embedding_dimension')}"
    )
if metadata.get("chunk_count") != len(vector_store):
    raise SystemExit(
        "Chunk count mismatch: "
        f"metadata has {metadata.get('chunk_count')}, index has {len(vector_store)}"
    )
if bool(metadata.get("pilot", False)) != pilot_enabled:
    raise SystemExit("Built index pilot status does not match configs/config.yaml")

print("Index build and validation completed successfully")
print(f"  path: {index_dir}")
print(f"  pilot: {pilot_enabled}")
print(f"  documents: {metadata.get('document_count')}")
print(f"  pages: {metadata.get('page_count')}")
print(f"  OCR words: {metadata.get('ocr_word_count')}")
print(f"  chunks: {len(vector_store)}")
print(f"  embedding model: {metadata.get('embedding_model')}")
print(f"  embedding dimension: {metadata.get('embedding_dimension')}")
print(f"  index type: {metadata.get('index_type')}")
PY
