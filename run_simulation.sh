#!/bin/bash

# Ensure script stops on error
set -e

echo "Installing dependencies..."
pip install -r requirements.txt

echo ""
echo "Running Discrete Mode Simulation (Example)..."
python simulate_drones.py --mode DISCRETE --episodes 10

echo ""
echo "Running Continuous Mode Simulation (Main Task)..."
python simulate_drones.py --mode CONTINUOUS --episodes 200

echo ""
echo "Done! Data saved to drone_data/"
