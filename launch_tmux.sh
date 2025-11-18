#!/bin/bash

# AI Camera Solution - Tmux Launcher Script
# This script creates separate tmux windows for monitoring different components

# Configuration
SESSION_NAME="aicamera"
CONFIG_FILE="${1:-config.yaml}"
OUTPUT_DIR="./output"
LOG_DIR="$OUTPUT_DIR/logs"

# Create directories
mkdir -p "$LOG_DIR"

# Check if tmux session already exists
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session $SESSION_NAME already exists. Killing it..."
    tmux kill-session -t "$SESSION_NAME"
fi

# Create new tmux session with main window
tmux new-session -d -s "$SESSION_NAME" -n "main"

# Window 0: Main application output
tmux send-keys -t "$SESSION_NAME:main" "cd '$(pwd)'" C-m
tmux send-keys -t "$SESSION_NAME:main" "echo '=== AI Camera Solution - Main Output ==='" C-m
tmux send-keys -t "$SESSION_NAME:main" "echo 'Running application...'" C-m
tmux send-keys -t "$SESSION_NAME:main" "./build/AICameraSolution --config '$CONFIG_FILE' 2>&1 | tee '$LOG_DIR/main.log'" C-m

# Window 1: Reader logs (filtered)
tmux new-window -t "$SESSION_NAME" -n "readers"
tmux send-keys -t "$SESSION_NAME:readers" "cd '$(pwd)'" C-m
tmux send-keys -t "$SESSION_NAME:readers" "echo '=== Reader Threads Log ==='" C-m
tmux send-keys -t "$SESSION_NAME:readers" "tail -f '$LOG_DIR/main.log' | grep --line-buffered '\[Reader\]'" C-m

# Window 2: Detector logs (filtered)
tmux new-window -t "$SESSION_NAME" -n "detectors"
tmux send-keys -t "$SESSION_NAME:detectors" "cd '$(pwd)'" C-m
tmux send-keys -t "$SESSION_NAME:detectors" "echo '=== Detector Threads Log ==='" C-m
tmux send-keys -t "$SESSION_NAME:detectors" "tail -f '$LOG_DIR/main.log' | grep --line-buffered '\[Detector\]'" C-m

# Window 3: Monitor/Statistics
tmux new-window -t "$SESSION_NAME" -n "monitor"
tmux send-keys -t "$SESSION_NAME:monitor" "cd '$(pwd)'" C-m
tmux send-keys -t "$SESSION_NAME:monitor" "echo '=== Statistics & Monitoring ==='" C-m
tmux send-keys -t "$SESSION_NAME:monitor" "tail -f '$LOG_DIR/main.log' | grep --line-buffered '\[Monitor\]\|\[STATS\]'" C-m

# Window 4: All logs (unfiltered)
tmux new-window -t "$SESSION_NAME" -n "all_logs"
tmux send-keys -t "$SESSION_NAME:all_logs" "cd '$(pwd)'" C-m
tmux send-keys -t "$SESSION_NAME:all_logs" "echo '=== All Logs (Unfiltered) ==='" C-m
tmux send-keys -t "$SESSION_NAME:all_logs" "tail -f '$LOG_DIR/main.log'" C-m

# Window 5: Control/Info
tmux new-window -t "$SESSION_NAME" -n "info"
tmux send-keys -t "$SESSION_NAME:info" "cd '$(pwd)'" C-m
tmux send-keys -t "$SESSION_NAME:info" "cat << 'EOF'
=== AI Camera Solution - Tmux Session ===

Session Name: $SESSION_NAME
Config File: $CONFIG_FILE
Output Directory: $OUTPUT_DIR
Log Directory: $LOG_DIR

Windows:
  0: main      - Main application output (running here)
  1: readers   - Reader thread logs (filtered)
  2: detectors - Detector thread logs (filtered)
  3: monitor   - Statistics and monitoring (filtered)
  4: all_logs  - All logs (unfiltered)
  5: info      - This information window

Navigation:
  - Switch windows: Ctrl+B then 0-5
  - Detach: Ctrl+B then D
  - Reattach: tmux attach -t $SESSION_NAME
  - Kill session: tmux kill-session -t $SESSION_NAME

Monitoring:
  - Statistics are printed every 5 seconds in the monitor window
  - All logs are saved to: $LOG_DIR/main.log
  - Processing log is also saved to: $OUTPUT_DIR/processing.log

Press any key to continue...
EOF
" C-m
tmux send-keys -t "$SESSION_NAME:info" "read -n 1" C-m

# Select the main window
tmux select-window -t "$SESSION_NAME:main"

echo ""
echo "=== Tmux Session Created ==="
echo "Session Name: $SESSION_NAME"
echo "Config File: $CONFIG_FILE"
echo ""
echo "Windows created:"
echo "  0: main      - Main application output"
echo "  1: readers   - Reader thread logs"
echo "  2: detectors - Detector thread logs"
echo "  3: monitor   - Statistics and monitoring"
echo "  4: all_logs  - All logs (unfiltered)"
echo "  5: info      - Information and help"
echo ""
echo "To attach to the session:"
echo "  tmux attach -t $SESSION_NAME"
echo ""
echo "To detach (while in tmux):"
echo "  Press: Ctrl+B, then D"
echo ""
echo "The application is starting in window 0 (main)..."
