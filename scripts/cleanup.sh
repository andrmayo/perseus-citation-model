#!/usr/bin/env bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "cleaning up files in rm $ROOT_DIR/model_data/"
rm "$ROOT_DIR/model_data/"*
echo "cleaning up files and directories in $ROOT_DIR/outputs/extraction/"
rm -rf "$ROOT_DIR/outputs/extraction/"*
echo "cleaning up files in $ROOT_DIR/outouts/models/extraction/"
rm "$ROOT_DIR/outputs/models/extraction/"*
