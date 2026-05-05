#!/bin/bash
# =============================================================================
# QA Generator Runner
# =============================================================================
# Runs the QA generator with .venv and proper monitoring.
#
# Usage:
#   ./run_qa_generator.sh              # Run in foreground
#   ./run_qa_generator.sh start        # Run in background
#   ./run_qa_generator.sh status       # Check status & progress
#   ./run_qa_generator.sh stop         # Stop background process
# =============================================================================

set -e
cd "$(dirname "$0")"
PROJECT_DIR="$(pwd)"

# Configuration
CONFIG_FILE="config/qa_generator.json"
VLLM_URL="http://localhost:8000/v1"
LOG_DIR="logs"
PID_FILE="$LOG_DIR/qa_generator.pid"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Activate virtual environment
activate_venv() {
    if [ -d ".venv" ]; then
        source .venv/bin/activate
        log_success "Activated .venv"
    else
        log_warn ".venv not found, using system Python"
    fi
}

check_vllm_server() {
    log_info "Checking vLLM server at $VLLM_URL..."
    
    if curl -s --max-time 5 "$VLLM_URL/models" > /dev/null 2>&1; then
        log_success "vLLM server is running"
        
        echo ""
        echo -e "${CYAN}Available models:${NC}"
        curl -s "$VLLM_URL/models" | python3 -c "
import sys, json
try:
    data = json.load(sys.stdin)
    for m in data.get('data', []):
        print(f\"  • {m['id']}\")
except:
    print('  (could not parse)')
" 2>/dev/null
        echo ""
        return 0
    else
        log_error "vLLM server not responding at $VLLM_URL"
        echo ""
        echo "Start vLLM first:"
        echo -e "  ${CYAN}./scripts/start_vllm_background.sh 0${NC}      # GPU 0 with 7B model"
        echo -e "  ${CYAN}./scripts/start_vllm_background.sh 2 32b${NC}  # GPU 2 with 32B model"
        echo ""
        return 1
    fi
}

check_input_files() {
    INPUT_DIR=$(python3 -c "import json; print(json.load(open('$CONFIG_FILE'))['input_dir'])" 2>/dev/null || echo "extracted_text/bulk_40k")
    
    if [ ! -d "$INPUT_DIR" ]; then
        log_error "Input directory not found: $INPUT_DIR"
        return 1
    fi
    
    FILE_COUNT=$(find "$INPUT_DIR" -name "*.txt" 2>/dev/null | wc -l)
    log_info "Found $FILE_COUNT .txt files in $INPUT_DIR"
    
    if [ "$FILE_COUNT" -eq 0 ]; then
        log_error "No .txt files found"
        return 1
    fi
    return 0
}

show_file_locations() {
    echo ""
    echo -e "${CYAN}File Locations:${NC}"
    echo "  Config:    $PROJECT_DIR/$CONFIG_FILE"
    echo "  Progress:  $PROJECT_DIR/data/qa_outputs/progress.json"
    echo "  Logs:      $PROJECT_DIR/data/qa_outputs/logs/"
    echo "  Output:    $PROJECT_DIR/data/qa_outputs/qa_by_file/"
    echo ""
}

show_progress() {
    PROGRESS_FILE="data/qa_outputs/progress.json"
    
    if [ -f "$PROGRESS_FILE" ]; then
        echo -e "${CYAN}Progress:${NC}"
        python3 -c "
import json
with open('$PROGRESS_FILE') as f:
    progress = json.load(f)
completed = sum(1 for v in progress.values() if v.get('completed', False))
in_progress = len(progress) - completed
total_qa = sum(v.get('qa_count', 0) for v in progress.values())
total_fail = sum(v.get('failure_count', 0) for v in progress.values())
print(f'  Files completed: {completed}')
print(f'  Files in progress: {in_progress}')
print(f'  Total QA pairs: {total_qa:,}')
print(f'  Total failures: {total_fail:,}')
" 2>/dev/null || echo "  (no progress yet)"
    else
        echo -e "${CYAN}Progress:${NC} No progress file yet"
    fi
}

show_gpu_status() {
    echo ""
    echo -e "${CYAN}GPU Status:${NC}"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | while read line; do
        echo "  $line"
    done || echo "  nvidia-smi not available"
}

run_foreground() {
    log_info "Starting QA Generator (foreground)..."
    echo ""
    
    python3 src/qa_generators/qa_generator.py \
        --config "$CONFIG_FILE" \
        "$@"
}

run_background() {
    mkdir -p "$LOG_DIR"
    
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUT_FILE="$LOG_DIR/qa_generator_${TIMESTAMP}.out"
    
    log_info "Starting QA Generator (background)..."
    
    # Run with venv activated
    nohup bash -c "
        cd '$PROJECT_DIR'
        source .venv/bin/activate 2>/dev/null || true
        python3 src/qa_generators/qa_generator.py --config '$CONFIG_FILE' $*
    " > "$OUT_FILE" 2>&1 &
    
    PID=$!
    echo $PID > "$PID_FILE"
    
    echo ""
    log_success "QA Generator started"
    echo ""
    echo "  PID: $PID"
    echo "  Log: $OUT_FILE"
    echo ""
    echo "Commands:"
    echo "  Monitor:  tail -f $OUT_FILE"
    echo "  Progress: ./run_qa_generator.sh status"
    echo "  Stop:     ./run_qa_generator.sh stop"
    echo ""
}

stop_generator() {
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            log_info "Stopping QA Generator (PID: $PID)..."
            kill $PID
            sleep 2
            if ps -p $PID > /dev/null 2>&1; then
                log_warn "Process still running, sending SIGKILL..."
                kill -9 $PID 2>/dev/null
            fi
            rm -f "$PID_FILE"
            log_success "Stopped"
        else
            log_warn "Process $PID not running"
            rm -f "$PID_FILE"
        fi
    else
        log_warn "No PID file found"
        # Try to find and kill any running instance
        pkill -f "qa_generator.py" 2>/dev/null && log_info "Killed orphaned process" || true
    fi
}

show_status() {
    echo ""
    echo "======================================"
    echo "       QA Generator Status"
    echo "======================================"
    
    # Check if running
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if ps -p $PID > /dev/null 2>&1; then
            log_success "Generator is RUNNING (PID: $PID)"
        else
            log_warn "Generator not running (stale PID file)"
        fi
    else
        log_info "Generator not running"
    fi
    
    # Check vLLM
    echo ""
    check_vllm_server || true
    
    # Show file locations
    show_file_locations
    
    # Show progress
    show_progress
    
    # Show GPU status
    show_gpu_status
    
    echo ""
}

show_help() {
    echo ""
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  run         Run in foreground (default)"
    echo "  start       Run in background"
    echo "  stop        Stop background process"
    echo "  status      Show status, progress, and GPU info"
    echo "  check       Verify vLLM server and input files"
    echo "  help        Show this help"
    echo ""
    echo "Options (passed to qa_generator.py):"
    echo "  --no-resume     Start fresh, ignore progress"
    echo "  --workers N     Number of parallel workers"
    echo "  --model NAME    Override model name"
    echo ""
    echo "Examples:"
    echo "  $0                        # Run in foreground"
    echo "  $0 start                  # Run in background"
    echo "  $0 start --workers 4      # Background, 4 workers"
    echo "  $0 status                 # Check progress"
    echo ""
    echo "Multi-GPU Setup:"
    echo "  # Start vLLM on GPU 0 (recommended for shared GPUs):"
    echo "  ./scripts/start_vllm_background.sh 0"
    echo ""
    echo "  # Or use GPU 2 with bigger model:"
    echo "  ./scripts/start_vllm_background.sh 2 32b"
    echo ""
    echo "  # Then run QA generator:"
    echo "  ./run_qa_generator.sh start"
    echo ""
}

# =============================================================================
# Main
# =============================================================================

mkdir -p "$LOG_DIR"
activate_venv

case "${1:-run}" in
    run)
        shift || true
        check_vllm_server || exit 1
        check_input_files || exit 1
        show_file_locations
        show_progress
        run_foreground "$@"
        ;;
    start)
        shift || true
        check_vllm_server || exit 1
        check_input_files || exit 1
        show_file_locations
        show_progress
        run_background "$@"
        ;;
    stop)
        stop_generator
        ;;
    status)
        show_status
        ;;
    check)
        check_vllm_server
        check_input_files
        show_file_locations
        show_progress
        show_gpu_status
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        log_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac
