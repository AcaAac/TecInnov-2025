#!/bin/bash

# Ensure script stops on error
set -e

# Get the script directory to navigate correctly
SCRIPT_DIR="$(dirname "$(realpath "$0")")"
# If realpath is missing (git bash usually has it), fallback:
# SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

# Navigate to project root
cd "$SCRIPT_DIR/.."

echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Running Full Pipeline (Collection -> BC -> RL -> Eval)..."
echo "Output will be saved to data/evasion_experiment/"

# Ensure output directory exists
mkdir -p data/evasion_experiment

# Run the training pipeline
# We assume python is in the path.
# We cd to src so imports work naturally (local imports in train_blue.py)
cd src

# Run with --all to execute the entire pipeline
python train_blue.py --all --output_dir ../data/evasion_experiment

echo ""
echo "Pipeline finished successfully! Check data/evasion_experiment/ for results."
