#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

tar -czvf "$ROOT_DIR/tarballs/cit_data.tar.gz" -C "$ROOT_DIR/cit_data" . 2>&1
tar -czvf "$ROOT_DIR/tarballs/model_data.tar.gz" -C "$ROOT_DIR/model_data" . 2>&1

TEMP=$(mktemp -d)
mkdir "$TEMP/extraction"
cp -r "$ROOT_DIR/outputs/extraction"/final-model* "$TEMP/extraction"
tar -czvf "$ROOT_DIR/tarballs/extraction_final_model.tar.gz" -C "$TEMP/extraction" . 2>&1
rm -rf "$TEMP"

echo "data for model compressed to tarballs/"
