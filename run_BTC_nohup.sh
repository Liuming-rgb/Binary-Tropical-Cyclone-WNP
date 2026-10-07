#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p BTC_out

nohup python -u BTC_FTIN_nohup.py > BTC_out/BTC_FTIN.log 2>&1 &
PID=$!
echo "$PID" > BTC_out/BTC_FTIN.pid

echo "BTC_FTIN started in background."
echo "PID: $PID"
echo "Log: BTC_out/BTC_FTIN.log"
echo "PID file: BTC_out/BTC_FTIN.pid"
echo "Watch log: tail -f BTC_out/BTC_FTIN.log"
