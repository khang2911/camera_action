#!/bin/bash

SESSION_NAME="aicamera"
CONFIG_FILE="${1:-config.yaml}"
OUTPUT_DIR="./output"
LOG_DIR="$OUTPUT_DIR/logs"

mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo "Session $SESSION_NAME already exists. Killing it..."
    tmux kill-session -t "$SESSION_NAME"
fi

# Window 0
tmux new-session -d -s "$SESSION_NAME" -n "main"

# Window 1
tmux new-window -t "$SESSION_NAME" -n "main-gpu2"

# Window 2
tmux new-window -t "$SESSION_NAME" -n "main-gpu3"

# Window 3
tmux new-window -t "$SESSION_NAME" -n "tracking"

# Window 4
tmux new-window -t "$SESSION_NAME" -n "detection"

# ---- GPU2 ----
tmux send-keys -t "$SESSION_NAME:main-gpu2" "cd '$(pwd)'" C-m
tmux send-keys -t "$SESSION_NAME:main-gpu2" "echo '=== AI Camera Solution - GPU2 ==='" C-m
tmux send-keys -t "$SESSION_NAME:main-gpu2" "CUDA_VISIBLE_DEVICES=GPU-1b5b0415-79c6-6cc7-3ceb-186c768cb88f ./build/detection --config '$CONFIG_FILE' | tee '$LOG_DIR/main_gpu2.log'" C-m

# ---- GPU3 ----
tmux send-keys -t "$SESSION_NAME:main-gpu3" "cd '$(pwd)'" C-m
tmux send-keys -t "$SESSION_NAME:main-gpu3" "echo '=== AI Camera Solution - GPU3 ==='" C-m
tmux send-keys -t "$SESSION_NAME:main-gpu3" "CUDA_VISIBLE_DEVICES=GPU-42025c82-fb33-1e84-7c25-8778ef907cf6 ./build/detection --config '$CONFIG_FILE' | tee '$LOG_DIR/main_gpu3.log'" C-m

# ---- Tracking ----
tmux send-keys -t "$SESSION_NAME:tracking" "cd '$(pwd)'" C-m
tmux send-keys -t "$SESSION_NAME:tracking" "python3 main_prod_tracking.py" C-m

# ---- Detection ----
tmux send-keys -t "$SESSION_NAME:detection" "cd '$(pwd)'" C-m
tmux send-keys -t "$SESSION_NAME:detection" "python3 main_prod_detection.py"

tmux select-window -t "$SESSION_NAME:main"

echo "Done. Attach with: tmux attach -t $SESSION_NAME"
