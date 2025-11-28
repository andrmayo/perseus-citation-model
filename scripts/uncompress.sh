#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

tar -xzvf "$ROOT_DIR/tarballs/model_data.tar.gz" --strip-components=1 -C "$ROOT_DIR/model_data"

tar -xzvf "$ROOT_DIR/tarballs/cit_data.tar.gz" --strip-components=1 -C "$ROOT_DIR/cit_data"

mkdir -p "$ROOT_DIR/outputs/extraction/from-tarball"
tar -xzvf "$ROOT_DIR/tarballs/extraction_final_model.tar.gz" --strip-components=1 -C "$ROOT_DIR/outputs/extraction/from-tarball"
mv "$ROOT_DIR/outputs/extraction/from-tarball/final-model"* "$ROOT_DIR/outputs/models/extraction"
rm -rf "$ROOT_DIR/outputs/extraction/from-tarball"

echo "files from tarballs/ extracted to their respective directories"
