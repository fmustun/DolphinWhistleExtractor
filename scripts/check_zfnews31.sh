#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-/media/zfnews31}"

echo "Target: $TARGET"
if mountpoint -q "$TARGET"; then
  echo "Mount: active"
else
  echo "Mount: inactive"
fi

if [ -d "$TARGET/Dolphins/Sound" ]; then
  echo "Path: $TARGET/Dolphins/Sound"
  find "$TARGET/Dolphins/Sound" -maxdepth 1 -type f | sed -n '1,10p'
else
  echo "Missing: $TARGET/Dolphins/Sound"
fi
