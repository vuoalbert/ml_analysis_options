#!/usr/bin/env bash
set -euo pipefail

DATA="${RENDER_DISK_PATH:-/var/data}"
mkdir -p "$DATA/artifacts" "$DATA/reports" "$DATA/logs" "$DATA/cache"

# Seed disk with image-baked artifacts on first boot
if [ -z "$(ls -A "$DATA/artifacts" 2>/dev/null)" ] && [ -d /app/artifacts ]; then
  cp -r /app/artifacts/. "$DATA/artifacts/"
fi

# Point in-image dirs at the persistent disk
for d in artifacts reports logs cache; do
  rm -rf "/app/$d"
  ln -s "$DATA/$d" "/app/$d"
done

# Background the trading loop; stream its logs to stdout so they show in Render
python -m live.loop --config v1 &
LOOP_PID=$!

# If the loop dies, kill the whole container so Render restarts it
( wait "$LOOP_PID"; echo "live.loop exited; shutting down"; kill 1 ) &

trap 'kill "$LOOP_PID" 2>/dev/null || true' EXIT

exec streamlit run ui/dashboard.py \
  --server.address=0.0.0.0 \
  --server.port="${PORT:-8501}"
