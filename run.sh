#!/usr/bin/env bash
set -euo pipefail

TRAIN_ITERS=200
EVAL_EPISODES=100
OUTPUT_DIR="./output_dir/7th_run"
VIDEO_EPISODES=5
# Slightly faster playback while retaining more visual detail than 20 FPS.
VIDEO_FPS=12


for arg in "$@"; do
  if [[ "$arg" == "--more-realistic" ]]; then
    ENV_CONFIG="config/more_realistic_env_config.json"
    break
  fi
done


python main.py \
  --train-iters "$TRAIN_ITERS" \
  --eval-episodes "$EVAL_EPISODES" \
  --output-dir "$OUTPUT_DIR" \
  --env-config "config/default_env_config.json" \
  --video \
  --video-episodes "$VIDEO_EPISODES" \
  --video-fps "$VIDEO_FPS" \
  --max-steps 1200 \
  "$@"
