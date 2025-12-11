#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

TEMP=$(mktemp -d)
mkdir "$TEMP/extraction"
cp -r "$ROOT_DIR/outputs/models/extraction/"* "$TEMP/extraction"
tar -czvf "$ROOT_DIR/tarballs/extraction_final_model.tar.gz" -C "$TEMP/extraction" . 2>&1
rm -rf "$TEMP"

echo "data for model compressed to tarballs/"
