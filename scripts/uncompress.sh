#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

tar -xzvf "$ROOT_DIR/tarballs/model_data.tar.gz" --strip-components=1 -C "$ROOT_DIR/model_data"

tar -xzvf "$ROOT_DIR/tarballs/cit_data.tar.gz" --strip-components=1 -C "$ROOT_DIR/cit_data"
