#!/bin/zsh
set -e

OUT_FILE="$1"

LOG_DIR="logs/$(date +%s)"
mkdir -p "$LOG_DIR"

cp $OUT_FILE "$LOG_DIR/output.log"
chmod +777 "$LOG_DIR/output.log"

cp -r /tmp/gcr_* "$LOG_DIR/"

chmod +777 "$LOG_DIR"
chmod +777 "$LOG_DIR"/*
