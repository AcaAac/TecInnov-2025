#!/bin/bash
set -e

# Configuration
MODE="CONTINUOUS"
CONFIG="configs/train.yaml"
MODEL_PATH="drone_data/rl_model_CONTINUOUS.pth"
EPISODES=10
VISUALIZE="True"

python src/Test_RL.py \
    --mode "$MODE" \
    --config "$CONFIG" \
    --model_path "$MODEL_PATH" \
    --episodes "$EPISODES" \
    --visualize "$VISUALIZE" \
    "$@"