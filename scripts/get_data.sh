#!/usr/bin/env bash
# A1 — fetch or recreate your scanned corpus into data/raw/
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

mkdir -p data/raw

if command -v gdown >/dev/null 2>&1; then
    GDOWN_BIN="$(command -v gdown)"
elif [[ -x ".venv/bin/gdown" ]]; then
    GDOWN_BIN=".venv/bin/gdown"
else
    echo "gdown is not installed in PATH or .venv. Install the project dependencies first." >&2
    exit 1
fi

PHYSICS_ID="13P5E08HcNbpALj8-aC-RUCy0d3iwodet"
CHEMISTRY_ID="1lbgge3uWpQmuHOBvlxJ4AuPP2Na1RYQ8"
BIOLOGY_ID="11WRMSWpqi4KtgxClDCBXely2XdiAhoju"

if [[ "$#" -eq 0 ]]; then
    REQUESTED_BOOKS=(physics chemistry biology)
else
    REQUESTED_BOOKS=("$@")
fi

for book in "${REQUESTED_BOOKS[@]}"; do
    case "$book" in
        physics|chemistry|biology) ;;
        *)
            echo "Unknown book '${book}'. Choose from: physics chemistry biology" >&2
            exit 2
            ;;
    esac
done

valid_pdf() {
    local path="$1"
    [[ -s "$path" ]] && [[ "$(head -c 5 "$path")" == "%PDF-" ]]
}

download_pdf() {
    local name="$1"
    local file_id="$2"
    local output="data/raw/${name}.pdf"
    local partial="data/raw/${name}.pdf.part"

    if valid_pdf "$output"; then
        echo "Using existing ${output}"
        return
    fi

    echo "Downloading ${name} textbook..."
    "$GDOWN_BIN" "$file_id" -O "$partial"
    if ! valid_pdf "$partial"; then
        echo "Downloaded file is not a valid PDF: ${partial}" >&2
        exit 1
    fi
    mv "$partial" "$output"
}

echo "Fetching corpus into data/raw/..."
for book in "${REQUESTED_BOOKS[@]}"; do
    case "$book" in
        physics) download_pdf "physics" "$PHYSICS_ID" ;;
        chemistry) download_pdf "chemistry" "$CHEMISTRY_ID" ;;
        biology) download_pdf "biology" "$BIOLOGY_ID" ;;
    esac
done

echo "Requested corpus successfully downloaded to data/raw/!"
