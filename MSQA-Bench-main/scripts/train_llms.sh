#!/bin/bash
#
# Background Training Script for LLM Fine-Tuning (QLoRA)
#
# Usage:
#   ./scripts/train_llms.sh start [--config path] [--subset N] [--gpu ID] [--model NAME]
#   ./scripts/train_llms.sh status
#   ./scripts/train_llms.sh logs
#   ./scripts/train_llms.sh stop
#   ./scripts/train_llms.sh summary
#   ./scripts/train_llms.sh compare
#
# Examples:
#   # Start training all models with default config on GPU 0
#   ./scripts/train_llms.sh start --gpu 0
#
#   # Train a single model
#   ./scripts/train_llms.sh start --model qwen2.5_3b --gpu 0
#
#   # Quick smoke test on 50 samples
#   ./scripts/train_llms.sh start --subset 50 --gpu 0
#

set -e

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${PROJECT_DIR}/logs"
PID_FILE="${LOG_DIR}/llm_training.pid"
ACTIVE_LOG_FILE="${LOG_DIR}/llm_training.logfile"
DEFAULT_CONFIG="${PROJECT_DIR}/config/llm_finetuner.json"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

mkdir -p "$LOG_DIR"

COMMAND="${1:-help}"
shift || true

CONFIG="$DEFAULT_CONFIG"
SUBSET=""
GPU_ID=""
MODEL_NAME=""
EXTRA_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --subset)
            SUBSET="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --model)
            MODEL_NAME="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

is_running() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

get_active_log() {
    if [ -f "$ACTIVE_LOG_FILE" ]; then
        ACTIVE=$(cat "$ACTIVE_LOG_FILE")
        if [ -f "$ACTIVE" ]; then
            echo "$ACTIVE"
            return
        fi
    fi
    ls -t "$LOG_DIR"/llm_training_*.log 2>/dev/null | head -1
}

start_training() {
    if is_running; then
        echo -e "${YELLOW}LLM training already running (PID: $(cat "$PID_FILE"))${NC}"
        echo "Use './scripts/train_llms.sh status' to check progress"
        exit 1
    fi

    if [ ! -f "$CONFIG" ]; then
        echo -e "${RED}Config file not found: $CONFIG${NC}"
        exit 1
    fi

    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="${LOG_DIR}/llm_training_${TIMESTAMP}.log"

    PYTHON="${PROJECT_DIR}/.venv/bin/python"
    if [ ! -f "$PYTHON" ]; then
        echo -e "${YELLOW}Warning: .venv not found, using system python${NC}"
        PYTHON="python"
    fi

    CMD="$PYTHON -u scripts/train_all_llms.py --config \"$CONFIG\""

    if [ -n "$SUBSET" ]; then
        CMD="$CMD --subset $SUBSET"
    fi
    if [ -n "$MODEL_NAME" ]; then
        CMD="$CMD --model $MODEL_NAME"
    fi
    CMD="$CMD $EXTRA_ARGS"

    echo -e "${GREEN}Starting LLM fine-tuning (QLoRA)...${NC}"
    echo "  Config: $CONFIG"
    echo "  Log file: $LOG_FILE"
    if [ -n "$SUBSET" ]; then
        echo "  Subset: $SUBSET examples"
    fi
    if [ -n "$MODEL_NAME" ]; then
        echo "  Model: $MODEL_NAME"
    fi
    if [ -n "$GPU_ID" ]; then
        echo "  GPU: $GPU_ID"
    fi
    echo ""

    cd "$PROJECT_DIR"

    if [ -n "$GPU_ID" ]; then
        export CUDA_VISIBLE_DEVICES="$GPU_ID"
        echo "Using GPU $GPU_ID (CUDA_VISIBLE_DEVICES=$GPU_ID)"
    fi

    # shellcheck disable=SC2086
    nohup bash -c "$CMD" > "$LOG_FILE" 2>&1 &
    PID=$!
    echo "$PID" > "$PID_FILE"
    echo "$LOG_FILE" > "$ACTIVE_LOG_FILE"

    echo -e "${GREEN}Training started!${NC}"
    echo "  PID: $PID"
    echo "  Log: $LOG_FILE"
}

stop_training() {
    if ! is_running; then
        echo -e "${YELLOW}No LLM training process found.${NC}"
        exit 0
    fi
    PID=$(cat "$PID_FILE")
    echo -e "${YELLOW}Stopping LLM training (PID: $PID)...${NC}"
    kill "$PID" 2>/dev/null || true
    rm -f "$PID_FILE"
    echo "Stop signal sent. Check logs to ensure graceful shutdown."
}

show_status() {
    if is_running; then
        PID=$(cat "$PID_FILE")
        echo -e "${GREEN}LLM training is running (PID: $PID)${NC}"
    else
        echo -e "${YELLOW}No active LLM training process.${NC}"
    fi
    LOG_FILE=$(get_active_log || true)
    if [ -n "$LOG_FILE" ]; then
        echo "Latest log file: $LOG_FILE"
    fi
    SUMMARY="${PROJECT_DIR}/logs/multi_llm_training_summary.json"
    if [ -f "$SUMMARY" ]; then
        echo "Last summary: $SUMMARY"
    fi
}

show_logs() {
    LOG_FILE=$(get_active_log)
    if [ -z "$LOG_FILE" ]; then
        echo -e "${YELLOW}No log file found yet.${NC}"
        exit 0
    fi
    echo -e "${BLUE}Tailing log: $LOG_FILE${NC}"
    tail -f "$LOG_FILE"
}

show_summary() {
    SUMMARY="${PROJECT_DIR}/logs/multi_llm_training_summary.json"
    if [ ! -f "$SUMMARY" ]; then
        echo -e "${YELLOW}No multi-model summary found at $SUMMARY${NC}"
        exit 0
    fi

    echo -e "${BLUE}Multi-model LLM training summary:${NC}"
    python - <<EOF
import json
from pathlib import Path

summary_path = Path("$SUMMARY")
data = json.loads(summary_path.read_text())
results = data.get("results", {})
if not results:
    print("No results recorded.")
    raise SystemExit(0)

print(f"Started at : {data.get('started_at')}")
print(f"Completed at: {data.get('completed_at')}")
print(f"Duration  s: {data.get('duration_seconds')}")
print("")
print(f"{'Model':<25} {'Status':<10} {'Adapter?':<8} {'Eval?':<6}  Log file")
print("-" * 80)

for name, info in results.items():
    status = info.get("status", "unknown")
    out_dir = Path(info.get("output_dir", f"models/fine_tuned_llms/{name}"))
    adapter_dir = out_dir / "final_adapter"
    eval_file = out_dir / "eval_results_test.json"
    adapter_ok = "yes" if adapter_dir.exists() else "no"
    eval_ok = "yes" if eval_file.exists() else "no"
    log_file = Path("logs") / f"llm_finetuning_{name}.log"
    print(f"{name:<25} {status:<10} {adapter_ok:<8} {eval_ok:<6}  {log_file}")
EOF
}

run_compare() {
    PYTHON="${PROJECT_DIR}/.venv/bin/python"
    if [ ! -f "$PYTHON" ]; then
        PYTHON="python"
    fi
    cd "$PROJECT_DIR"
    $PYTHON -u -c "from src.llm_trainers.model_comparison import compare_from_directory; compare_from_directory('models/fine_tuned_llms')" || true
}

case "$COMMAND" in
    start)
        start_training
        ;;
    stop)
        stop_training
        ;;
    status)
        show_status
        ;;
    logs)
        show_logs
        ;;
    summary)
        show_summary
        ;;
    compare)
        run_compare
        ;;
    *)
        echo "Usage: $0 {start|stop|status|logs|summary|compare} [options]"
        exit 1
        ;;
esac

