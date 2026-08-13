#!/usr/bin/env bash
# A1 — fetch or recreate your scanned corpus into data/raw/
# IMPLEMENT: download/prepare your corpus (public-domain / openly licensed).
set -euo pipefail

# Create the data/raw directory if it doesn't exist
mkdir -p data/raw

# We use gdown to handle Google Drive links (which block standard curl/wget for large files)
if ! command -v gdown &> /dev/null; then
    echo "gdown is not installed. Installing via pip..."
    pip install gdown
fi

echo "Fetching corpus into data/raw/..."

# Physics: Google Drive File ID extracted from your link
PHYSICS_ID="13P5E08HcNbpALj8-aC-RUCy0d3iwodet"
CHEMISTRY_ID="1lbgge3uWpQmuHOBvlxJ4AuPP2Na1RYQ8"
BIOLOGY_ID="11WRMSWpqi4KtgxClDCBXely2XdiAhoju"

echo "Downloading Physics textbook..."
gdown "$PHYSICS_ID" -O data/raw/physics.pdf

echo "Downloading Chemistry textbook..."
gdown "$CHEMISTRY_ID" -O data/raw/chemistry.pdf

echo "Downloading Biology textbook..."
gdown "$BIOLOGY_ID" -O data/raw/biology.pdf

echo "Corpus successfully downloaded to data/raw/!"
