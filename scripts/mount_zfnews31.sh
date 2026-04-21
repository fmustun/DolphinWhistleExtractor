#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-/media/zfnews31}"
REMOTE="${ZFNEWS31_REMOTE:-pablo@ens-ibens:/media/zfnews31}"
MOUNT_GID="${ZFNEWS31_GID:-1038}"

mkdir -p "$TARGET"

if mountpoint -q "$TARGET"; then
  echo "$TARGET is already mounted."
  exit 0
fi

sshfs \
  -o "reconnect,ServerAliveInterval=15,ServerAliveCountMax=3,uid=$(id -u),gid=${MOUNT_GID}" \
  "$REMOTE" \
  "$TARGET"

echo "Mounted $REMOTE at $TARGET"
echo "Try: ls \"$TARGET/Dolphins/Sound\" | head"
