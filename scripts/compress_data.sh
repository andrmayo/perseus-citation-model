#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

tar -czvf "$ROOT_DIR/tarballs/cit_data.tar.gz" -C "$ROOT_DIR/cit_data" . 2>&1
tar -czvf "$ROOT_DIR/tarballs/model_data.tar.gz" -C "$ROOT_DIR/model_data" . 2>&1
