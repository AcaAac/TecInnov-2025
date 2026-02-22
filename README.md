# Drone RL Training Guide

I have implemented an Imitation Learning (BC) and Reinforcement Learning (RL) pipeline for the Blue agent. This project simulates a "tag" game where a Blue drone learns to evade a Red pursuer.

## New Files
- [train_blue.py](file:///train_blue.py): Main script for data collection, BC, and RL.
- [drone_env.py](file:///drone_env.py): Refactored environment module (shared by sim and trainer).
- [ppo.py](file:///ppo.py): PyTorch implementation of PPO (Proximal Policy Optimization) and Neural Network definitions ([ActorCritic](file:///ppo.py#10-69)).

## Implementation Details
1.  **Expert Demo Collection**: Runs the heuristic [BlueEvasivePolicy](file:///drone_env.py) to generate a dataset.
2.  **Behavior Cloning (BC)**: Pre-trains the neural network to mimic the expert's actions.
3.  **RL Fine-tuning**: Uses PPO to optimize the pre-trained policy against the Red pursuer, maximizing survival time and distance.

**Key Features:**
*   **Normalized Inputs**: States (position/velocity) are normalized to `[-1, 1]` for stable training.
*   **BC Monitoring**: Behavior cloning now logs train/validation metrics and saves dedicated BC training plots.
*   **Richer RL Diagnostics**: PPO now logs return, catch-rate, distance, policy/value losses, entropy, KL, and clip fraction.
*   **Live Training Plots**: Optional `--live_plots` for non-blocking plot updates during BC/RL training.
*   **Visualization**: Real-time rendering of the agents during evaluation.

## How to Run

### Prerequisite
You need **PyTorch** and **Matplotlib** installed.
```bash
create an environment and install the following packages:

conda create -n drone_env python=3.8
conda activate drone_env
pip install -r requirements.txt

or 
pip install torch matplotlib pandas numpy
```

### Full Pipeline (Recommended)
This runs all steps in order:
```bash
python src/train_blue.py --mode CONTINUOUS --all --episodes_demo 2000 --steps_rl 100000 --bc_epochs 30 --steps_per_epoch 2048 --visualize
```

### Step-by-Step

**1. Collect Demonstrations**
*Required first step due to input normalization.*
```bash
python src/train_blue.py --mode CONTINUOUS --collect_demos --episodes_demo 2000
```
*Output: `drone_data/expert_CONTINUOUS.pt`*

**2. Train Behavior Cloning**
```bash
python src/train_blue.py --mode CONTINUOUS --train_bc --bc_epochs 30
```
*Output:*
- *Model: `drone_data/bc_model_CONTINUOUS.pth`*
- *Logs: `drone_data/bc_training_log_CONTINUOUS.csv`*
- *Plot: `drone_data/bc_training_plot_CONTINUOUS.png`*

**3. Fine-tune with RL**
```bash
python src/train_blue.py --mode CONTINUOUS --train_rl --steps_rl 200000 --steps_per_epoch 2048
```
*Output:*
- *Model: `drone_data/rl_model_CONTINUOUS.pth`*
- *Logs: `drone_data/training_log_CONTINUOUS.csv`*
- *Plot: `drone_data/rl_training_plot_CONTINUOUS.png`*

**4. Evaluate Results**
```bash
python src/train_blue.py --mode CONTINUOUS --eval --visualize
```
This will run the **Expert**, **BC**, and **BC+RL** agents against Red and save a comparison CSV to `drone_data/evaluation_results_CONTINUOUS.csv`.
*   Adds one flag `--visualize` to see the agents in action (matplotlib window).

### Helpful Flags

```bash
# Show live BC/RL plot updates while training
python src/train_blue.py --mode CONTINUOUS --train_bc --train_rl --live_plots
```

```bash
# Faster iteration for debugging
python src/train_blue.py --mode CONTINUOUS --all --episodes_demo 200 --bc_epochs 10 --steps_rl 20000 --steps_per_epoch 1024
```

> **Note on Windows/DLL Errors:** If you encounter `ImportError: DLL load failed` with PyTorch, you may need to install the [Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist) or try installing `intel-openmp`. The data collection step (`--collect_demos`) will work even without PyTorch.
