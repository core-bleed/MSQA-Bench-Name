#!/bin/bash
# Run MSQA-Bench paper pipeline in background (for server use).
# Saves stdout/stderr to paper_results/ and continues after disconnect.
#
# Usage:
#   ./scripts/run_paper_pipeline_background.sh
#   ./scripts/run_paper_pipeline_background.sh --steps dataset
#
# Monitor: tail -f paper_results/pipeline_background.log
# Check status: ps aux | grep run_paper_pipeline

set -e
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
RESULTS="${ROOT}/paper_results"
mkdir -p "$RESULTS"
LOG="${RESULTS}/pipeline_background.log"
TS=$(date +%Y%m%d_%H%M%S)
echo "[$TS] Starting paper pipeline (background)" >> "$LOG"
echo "[$TS] Output directory: $RESULTS" >> "$LOG"
echo "[$TS] Command: python scripts/run_paper_pipeline.py $*" >> "$LOG"
nohup python scripts/run_paper_pipeline.py --config config/paper_pipeline.json "$@" >> "$LOG" 2>&1 &
PID=$!
echo "[$TS] PID: $PID" >> "$LOG"
echo "$PID" > "${RESULTS}/pipeline.pid"
echo "Pipeline started in background (PID $PID)"
echo "  Log: tail -f $LOG"
echo "  Stop: kill $PID"
