#!/bin/bash
#
# Background Training Script for Embedding Fine-Tuning
#
# Usage:
#   ./scripts/train_embeddings.sh start [--config path] [--subset N] [--gpu ID]
#   ./scripts/train_embeddings.sh stop
#   ./scripts/train_embeddings.sh status
#   ./scripts/train_embeddings.sh logs
#   ./scripts/train_embeddings.sh resume [--gpu ID]
#
# Examples:
#   # Start training with default config
#   ./scripts/train_embeddings.sh start
#
#   # Start with custom config
#   ./scripts/train_embeddings.sh start --config config/custom.json
#
#   # Start with subset for testing
#   ./scripts/train_embeddings.sh start --subset 10000
#
#   # Start on specific GPU (0, 1, or 2)
#   ./scripts/train_embeddings.sh start --gpu 2
#
#   # Resume from checkpoint on GPU 2
#   ./scripts/train_embeddings.sh resume --gpu 2
#
#   # Check status
#   ./scripts/train_embeddings.sh status
#
#   # View live logs
#   ./scripts/train_embeddings.sh logs
#
#   # Stop training gracefully
#   ./scripts/train_embeddings.sh stop
#

set -e

# Ensure CUDA device ordering matches nvidia-smi indices (by PCI bus id).
# Without this, CUDA device ordinals may not match `nvidia-smi` index numbers,
# so CUDA_VISIBLE_DEVICES=2 might not be the RTX 3090 Ti.
export CUDA_DEVICE_ORDER=PCI_BUS_ID

# Help PyTorch reduce CUDA memory fragmentation (recommended in OOM message)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
LOG_DIR="${PROJECT_DIR}/logs"
PID_FILE="${LOG_DIR}/training.pid"
DEFAULT_CONFIG="${PROJECT_DIR}/config/embedding_finetuner.json"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Create log directory
mkdir -p "$LOG_DIR"

# Parse command
COMMAND="${1:-help}"
shift || true

# Parse additional arguments
CONFIG="$DEFAULT_CONFIG"
SUBSET=""
GPU_ID="2"  # Default to GPU 2
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
        *)
            EXTRA_ARGS="$EXTRA_ARGS $1"
            shift
            ;;
    esac
done

# Function to check if training is running
is_running() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            return 0
        fi
    fi
    return 1
}

# Function to get latest log file
get_latest_log() {
    # Check for active training log first
    if [ -f "${LOG_DIR}/training.logfile" ]; then
        ACTIVE_LOG=$(cat "${LOG_DIR}/training.logfile")
        if [ -f "$ACTIVE_LOG" ]; then
            echo "$ACTIVE_LOG"
            return
        fi
    fi
    # Fallback to latest log
    ls -t "$LOG_DIR"/training*.log 2>/dev/null | head -1
}

# Function to show GPU info
show_gpu_info() {
    if command -v nvidia-smi &> /dev/null; then
        echo -e "${BLUE}GPU Status:${NC}"
        nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits | \
            awk -F',' '{printf "  %s - Memory: %s/%s MB (%s%% utilized)\n", $1, $2, $3, $4}'
    fi
}

# Start training
start_training() {
    if is_running; then
        echo -e "${YELLOW}Training is already running (PID: $(cat $PID_FILE))${NC}"
        echo "Use './scripts/train_embeddings.sh status' to check progress"
        echo "Use './scripts/train_embeddings.sh stop' to stop current training"
        exit 1
    fi
    
    # Check if config exists
    if [ ! -f "$CONFIG" ]; then
        echo -e "${RED}Config file not found: $CONFIG${NC}"
        exit 1
    fi
    
    # Generate log filename
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="${LOG_DIR}/training_${TIMESTAMP}.log"
    
    # Build command
    # Use Python from virtual environment
    PYTHON="${PROJECT_DIR}/.venv/bin/python"
    if [ ! -f "$PYTHON" ]; then
        echo -e "${YELLOW}Warning: .venv not found, using system python${NC}"
        PYTHON="python"
    fi
    
    CMD="$PYTHON -u -m src.embedding_trainers.streaming_finetuner --config $CONFIG"
    
    if [ -n "$SUBSET" ]; then
        CMD="$CMD --subset_size $SUBSET"
    fi
    
    CMD="$CMD $EXTRA_ARGS"
    
    echo -e "${GREEN}Starting embedding fine-tuning...${NC}"
    echo "  Config: $CONFIG"
    echo "  Log file: $LOG_FILE"
    if [ -n "$SUBSET" ]; then
        echo "  Subset: $SUBSET examples"
    fi
    if [ -n "$GPU_ID" ]; then
        echo "  GPU: $GPU_ID"
    fi
    echo ""
    
    show_gpu_info
    echo ""
    
    # Start training in background
    cd "$PROJECT_DIR"
    
    # Set GPU if specified
    if [ -n "$GPU_ID" ]; then
        export CUDA_VISIBLE_DEVICES="$GPU_ID"
        echo "Using GPU $GPU_ID (CUDA_VISIBLE_DEVICES=$GPU_ID)"
    fi
    
    nohup env CUDA_VISIBLE_DEVICES="${GPU_ID:-}" $CMD > "$LOG_FILE" 2>&1 &
    PID=$!
    echo $PID > "$PID_FILE"
    
    # Save GPU info for status command
    echo "$GPU_ID" > "${LOG_DIR}/training.gpu"
    
    echo -e "${GREEN}Training started!${NC}"
    echo "  PID: $PID"
    echo "  Log: $LOG_FILE"
    if [ -n "$GPU_ID" ]; then
        echo "  GPU: $GPU_ID"
    fi
    echo ""
    echo "Commands:"
    echo "  ./scripts/train_embeddings.sh status  - Check progress"
    echo "  ./scripts/train_embeddings.sh logs    - View live logs"
    echo "  ./scripts/train_embeddings.sh stop    - Stop training"
    
    # Show initial log output
    echo ""
    echo -e "${BLUE}Initial output (Ctrl+C to detach):${NC}"
    sleep 2
    tail -f "$LOG_FILE" &
    TAIL_PID=$!
    
    # Wait a bit then kill tail
    sleep 5
    kill $TAIL_PID 2>/dev/null || true
    echo ""
    echo -e "${GREEN}Training is running in background.${NC}"
}

# Resume training
resume_training() {
    if is_running; then
        echo -e "${YELLOW}Training is already running (PID: $(cat $PID_FILE))${NC}"
        exit 1
    fi
    
    # Check for checkpoints
    CHECKPOINT_DIR="${PROJECT_DIR}/models/fine_tuned_embeddings/checkpoints"
    if [ ! -d "$CHECKPOINT_DIR" ] || [ -z "$(ls -A $CHECKPOINT_DIR 2>/dev/null)" ]; then
        echo -e "${RED}No checkpoints found to resume from${NC}"
        echo "Start new training with: ./scripts/train_embeddings.sh start"
        exit 1
    fi
    
    # Generate log filename
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="${LOG_DIR}/training_resume_${TIMESTAMP}.log"
    
    # Build command with --resume flag
    # Use Python from virtual environment
    PYTHON="${PROJECT_DIR}/.venv/bin/python"
    if [ ! -f "$PYTHON" ]; then
        echo -e "${YELLOW}Warning: .venv not found, using system python${NC}"
        PYTHON="python"
    fi
    
    CMD="$PYTHON -u -m src.embedding_trainers.streaming_finetuner --config $CONFIG --resume"
    CMD="$CMD $EXTRA_ARGS"
    
    echo -e "${GREEN}Resuming training from checkpoint...${NC}"
    echo "  Config: $CONFIG"
    echo "  Log file: $LOG_FILE"
    if [ -n "$GPU_ID" ]; then
        echo "  GPU: $GPU_ID"
    fi
    echo ""
    
    # Start training in background
    cd "$PROJECT_DIR"
    
    # Set GPU if specified
    if [ -n "$GPU_ID" ]; then
        export CUDA_VISIBLE_DEVICES="$GPU_ID"
        echo "Using GPU $GPU_ID (CUDA_VISIBLE_DEVICES=$GPU_ID)"
    fi
    
    nohup env CUDA_VISIBLE_DEVICES="${GPU_ID:-}" $CMD > "$LOG_FILE" 2>&1 &
    PID=$!
    echo $PID > "$PID_FILE"
    
    # Save GPU info for status command
    echo "$GPU_ID" > "${LOG_DIR}/training.gpu"
    
    echo -e "${GREEN}Training resumed!${NC}"
    echo "  PID: $PID"
    echo "  Log: $LOG_FILE"
    if [ -n "$GPU_ID" ]; then
        echo "  GPU: $GPU_ID"
    fi
}

# Stop training
stop_training() {
    if ! is_running; then
        echo -e "${YELLOW}Training is not running${NC}"
        # Clean up stale PID file
        rm -f "$PID_FILE"
        exit 0
    fi
    
    PID=$(cat "$PID_FILE")
    echo -e "${YELLOW}Stopping training (PID: $PID)...${NC}"
    echo "Sending SIGTERM for graceful shutdown..."
    
    kill -SIGTERM "$PID" 2>/dev/null || true
    
    # Wait for graceful shutdown
    echo "Waiting for checkpoint save..."
    for i in {1..30}; do
        if ! kill -0 "$PID" 2>/dev/null; then
            echo -e "${GREEN}Training stopped gracefully${NC}"
            rm -f "$PID_FILE"
            
            # Show checkpoint info
            LATEST_LOG=$(get_latest_log)
            if [ -n "$LATEST_LOG" ]; then
                echo ""
                echo "Last log entries:"
                tail -5 "$LATEST_LOG"
            fi
            
            echo ""
            echo "To resume: ./scripts/train_embeddings.sh resume"
            exit 0
        fi
        sleep 1
        echo -n "."
    done
    
    # Force kill if still running
    echo ""
    echo -e "${RED}Process didn't stop gracefully, forcing...${NC}"
    kill -9 "$PID" 2>/dev/null || true
    rm -f "$PID_FILE"
    echo "Training killed"
}

# Show status
show_status() {
    echo -e "${BLUE}=== Training Status ===${NC}"
    echo ""
    
    if is_running; then
        PID=$(cat "$PID_FILE")
        echo -e "Status: ${GREEN}RUNNING${NC} (PID: $PID)"
        
        # Show GPU assignment
        GPU_FILE="${LOG_DIR}/training.gpu"
        if [ -f "$GPU_FILE" ] && [ -s "$GPU_FILE" ]; then
            ASSIGNED_GPU=$(cat "$GPU_FILE")
            echo -e "GPU: ${GREEN}$ASSIGNED_GPU${NC}"
        fi
        
        # Show process info
        echo ""
        echo "Process info:"
        ps -p $PID -o pid,ppid,%cpu,%mem,etime,cmd --no-headers 2>/dev/null | head -1
        
        # Show GPU usage
        echo ""
        show_gpu_info
        
        # Show recent log entries
        LATEST_LOG=$(get_latest_log)
        if [ -n "$LATEST_LOG" ]; then
            echo ""
            echo -e "${BLUE}Recent log entries:${NC}"
            tail -10 "$LATEST_LOG"
        fi
        
        # Show progress file if exists
        PROGRESS_FILE="${PROJECT_DIR}/models/fine_tuned_embeddings/training_progress.json"
        if [ -f "$PROGRESS_FILE" ]; then
            echo ""
            echo -e "${BLUE}Training progress:${NC}"
            cat "$PROGRESS_FILE" | python -m json.tool 2>/dev/null || cat "$PROGRESS_FILE"
        fi
    else
        echo -e "Status: ${YELLOW}NOT RUNNING${NC}"
        
        # Check for completed training
        FINAL_MODEL="${PROJECT_DIR}/models/fine_tuned_embeddings/final_model"
        if [ -d "$FINAL_MODEL" ]; then
            echo ""
            echo -e "${GREEN}Final model found: $FINAL_MODEL${NC}"
            
            # Show summary if exists
            SUMMARY="${PROJECT_DIR}/models/fine_tuned_embeddings/training_summary.json"
            if [ -f "$SUMMARY" ]; then
                echo ""
                echo "Training summary:"
                cat "$SUMMARY" | python -m json.tool 2>/dev/null || cat "$SUMMARY"
            fi
        fi
        
        # Check for checkpoints
        CHECKPOINT_DIR="${PROJECT_DIR}/models/fine_tuned_embeddings/checkpoints"
        if [ -d "$CHECKPOINT_DIR" ] && [ -n "$(ls -A $CHECKPOINT_DIR 2>/dev/null)" ]; then
            echo ""
            echo -e "${YELLOW}Checkpoints available:${NC}"
            ls -la "$CHECKPOINT_DIR"
            echo ""
            echo "Resume with: ./scripts/train_embeddings.sh resume"
        fi
        
        # Clean up stale PID file
        rm -f "$PID_FILE"
    fi
}

# Show logs
show_logs() {
    LATEST_LOG=$(get_latest_log)
    
    if [ -z "$LATEST_LOG" ]; then
        echo -e "${YELLOW}No log files found${NC}"
        exit 1
    fi
    
    echo -e "${BLUE}Showing logs from: $LATEST_LOG${NC}"
    echo "Press Ctrl+C to exit"
    echo ""
    
    tail -f "$LATEST_LOG"
}

# Train all models
train_all_models() {
    # Force all-model training to use GPU 2 (RTX 3090 Ti)
    GPU_ID="2"

    if is_running; then
        echo -e "${YELLOW}Training is already running (PID: $(cat $PID_FILE))${NC}"
        echo "Use './scripts/train_embeddings.sh status' to check progress"
        echo "Use './scripts/train_embeddings.sh stop' to stop current training"
        exit 1
    fi
    
    # Check if config exists
    if [ ! -f "$CONFIG" ]; then
        echo -e "${RED}Config file not found: $CONFIG${NC}"
        exit 1
    fi
    
    # Generate log filename
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    LOG_FILE="${LOG_DIR}/training_all_models_${TIMESTAMP}.log"
    
    # Build command
    PYTHON="${PROJECT_DIR}/.venv/bin/python"
    if [ ! -f "$PYTHON" ]; then
        echo -e "${YELLOW}Warning: .venv not found, using system python${NC}"
        PYTHON="python"
    fi
    
    CMD="$PYTHON -u scripts/train_all_models.py --config $CONFIG --yes"
    
    if [ -n "$SUBSET" ]; then
        CMD="$CMD --subset $SUBSET"
    fi
    
    if [ -n "$GPU_ID" ]; then
        CMD="$CMD --gpu $GPU_ID"
    fi
    
    # Check for --cpu flag in extra args
    if [[ "$EXTRA_ARGS" == *"--cpu"* ]]; then
        CMD="$CMD --cpu"
    fi
    
    CMD="$CMD $EXTRA_ARGS"
    
    echo -e "${GREEN}Starting multi-model training...${NC}"
    echo "  Config: $CONFIG"
    echo "  Log file: $LOG_FILE"
    if [ -n "$SUBSET" ]; then
        echo "  Subset: $SUBSET examples"
    fi
    if [[ "$EXTRA_ARGS" == *"--cpu"* ]]; then
        echo "  Mode: CPU (forced)"
    elif [ -n "$GPU_ID" ]; then
        echo "  GPU: $GPU_ID"
    fi
    echo ""
    
    show_gpu_info
    echo ""
    
    # Start training in background
    cd "$PROJECT_DIR"
    
    # Set GPU/CPU environment
    if [[ "$EXTRA_ARGS" == *"--cpu"* ]]; then
        export CUDA_VISIBLE_DEVICES=""
        echo "Forcing CPU usage (CUDA_VISIBLE_DEVICES='')"
        ENV_VARS="CUDA_VISIBLE_DEVICES="
    elif [ -n "$GPU_ID" ]; then
        export CUDA_VISIBLE_DEVICES="$GPU_ID"
        echo "Using GPU $GPU_ID (CUDA_VISIBLE_DEVICES=$GPU_ID)"
        ENV_VARS="CUDA_VISIBLE_DEVICES=$GPU_ID"
    else
        ENV_VARS=""
    fi
    
    if [ -n "$ENV_VARS" ]; then
        nohup env $ENV_VARS $CMD > "$LOG_FILE" 2>&1 &
    else
        nohup $CMD > "$LOG_FILE" 2>&1 &
    fi
    PID=$!
    echo $PID > "$PID_FILE"
    
    # Save GPU info for status command
    if [[ "$EXTRA_ARGS" == *"--cpu"* ]]; then
        echo "cpu" > "${LOG_DIR}/training.gpu"
    else
        echo "$GPU_ID" > "${LOG_DIR}/training.gpu"
    fi
    
    # Save log file path for status command
    echo "$LOG_FILE" > "${LOG_DIR}/training.logfile"
    
    echo -e "${GREEN}Multi-model training started!${NC}"
    echo "  PID: $PID"
    echo "  Log: $LOG_FILE"
    if [ -n "$GPU_ID" ]; then
        echo "  GPU: $GPU_ID"
    fi
    echo ""
    echo "Commands:"
    echo "  ./scripts/train_embeddings.sh status  - Check progress"
    echo "  ./scripts/train_embeddings.sh logs    - View live logs"
    echo "  ./scripts/train_embeddings.sh stop    - Stop training"
    
    # Show initial log output
    echo ""
    echo -e "${BLUE}Initial output (Ctrl+C to detach):${NC}"
    sleep 2
    tail -f "$LOG_FILE" &
    TAIL_PID=$!
    
    # Wait a bit then kill tail
    sleep 5
    kill $TAIL_PID 2>/dev/null || true
    echo ""
    echo -e "${GREEN}Training is running in background.${NC}"
}

# Show help
show_help() {
    echo "Embedding Fine-Tuning Training Script"
    echo ""
    echo "Usage:"
    echo "  ./scripts/train_embeddings.sh <command> [options]"
    echo ""
    echo "Commands:"
    echo "  start   Start training in background"
    echo "  stop    Stop training gracefully (saves checkpoint)"
    echo "  status  Show training status and progress"
    echo "  logs    Show live training logs"
    echo "  resume  Resume training from checkpoint"
    echo "  all     Train all models from config (sequential)"
    echo "  help    Show this help message"
    echo ""
    echo "Options:"
    echo "  --config PATH    Path to config file (default: config/embedding_finetuner.json)"
    echo "  --subset N       Train on first N examples (for testing)"
    echo "  --gpu ID         Use specific GPU (0, 1, 2, etc.)"
    echo ""
    echo "Examples:"
    echo "  # Start training single model"
    echo "  ./scripts/train_embeddings.sh start"
    echo ""
    echo "  # Train all models from config"
    echo "  ./scripts/train_embeddings.sh all"
    echo ""
    echo "  # Train all models with subset for testing"
    echo "  ./scripts/train_embeddings.sh all --subset 10000"
    echo ""
    echo "  # Train all models on specific GPU"
    echo "  ./scripts/train_embeddings.sh all --gpu 2"
    echo ""
    echo "  # Start with subset for quick test"
    echo "  ./scripts/train_embeddings.sh start --subset 10000"
    echo ""
    echo "  # Start on specific GPU (e.g., GPU 2)"
    echo "  ./scripts/train_embeddings.sh start --gpu 2"
    echo ""
    echo "  # Resume from checkpoint after crash/stop"
    echo "  ./scripts/train_embeddings.sh resume --gpu 2"
    echo ""
    echo "  # Check progress"
    echo "  ./scripts/train_embeddings.sh status"
}

# Main command dispatcher
case $COMMAND in
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
    resume)
        resume_training
        ;;
    all)
        train_all_models
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: $COMMAND${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac
