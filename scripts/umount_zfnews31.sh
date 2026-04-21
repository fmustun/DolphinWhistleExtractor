#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-/media/zfnews31}"

if ! mountpoint -q "$TARGET"; then
  echo "$TARGET is not mounted."
  exit 0
fi

fusermount3 -uz "$TARGET" 2>/dev/null || fusermount -uz "$TARGET"

echo "Unmounted $TARGET"
