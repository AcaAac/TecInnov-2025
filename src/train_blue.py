import argparse
import copy
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset

    from ppo import PPOAgent

    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    print("Warning: PyTorch not found. Training will not work, but data collection might.")
    torch = None
    DEVICE = None

from env import BlueEvasivePolicy, DroneEnv, load_env_config


def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)


def moving_average(values, window=10):
    arr = np.asarray(values, dtype=np.float32)
    if arr.size == 0:
        return arr
    if window <= 1 or arr.size < window:
        return arr
    kernel = np.ones(window, dtype=np.float32) / window
    smoothed = np.convolve(arr, kernel, mode="valid")
    padding = np.full(window - 1, np.nan, dtype=np.float32)
    return np.concatenate([padding, smoothed])


def _ensure_numpy(a):
    if torch is not None and isinstance(a, torch.Tensor):
        return a.detach().cpu().numpy()
    return np.asarray(a)


def _bc_batch_metrics(agent, bx, by, mode, criterion, cfg):
    trunk = agent.model.trunk(bx)

    if mode == "DISCRETE":
        logits = agent.model.actor(trunk)
        loss = criterion(logits, by)
        accuracy = (logits.argmax(dim=-1) == by).float().mean()
        return loss, accuracy

    preds = torch.tanh(agent.model.actor_mean(trunk))
    target = torch.clamp(by / cfg.BLUE_MAX_ACCEL, -1.0, 1.0)
    loss = criterion(preds, target)

    pred_action = preds * cfg.BLUE_MAX_ACCEL
    mae = torch.mean(torch.abs(pred_action - by))
    return loss, mae


def _update_bc_live_plot(fig, axes, log_df, mode):
    epochs = log_df["epoch"].to_numpy()

    axes[0].cla()
    axes[0].plot(epochs, log_df["train_loss"], label="Train Loss", color="tab:blue")
    axes[0].plot(epochs, log_df["val_loss"], label="Val Loss", color="tab:orange")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(f"BC Loss ({mode})")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].cla()
    if "train_accuracy" in log_df.columns:
        axes[1].plot(epochs, log_df["train_accuracy"], label="Train Acc", color="tab:green")
        axes[1].plot(epochs, log_df["val_accuracy"], label="Val Acc", color="tab:red")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_ylim(0.0, 1.0)
    else:
        axes[1].plot(epochs, log_df["train_action_mae"], label="Train MAE", color="tab:green")
        axes[1].plot(epochs, log_df["val_action_mae"], label="Val MAE", color="tab:red")
        axes[1].set_ylabel("Action MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_title(f"BC Metric ({mode})")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.001)


def plot_bc_training_results(log_df, output_dir, mode):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    _update_bc_live_plot(fig, axes, log_df, mode)

    save_path = os.path.join(output_dir, f"bc_training_plot_{mode}.png")
    fig.savefig(save_path)
    print(f"BC training plot saved to {save_path}")
    plt.close(fig)


def _update_rl_live_plot(fig, axes, log_df, mode):
    epochs = log_df["epoch"].to_numpy()

    axes[0, 0].cla()
    axes[0, 0].plot(epochs, log_df["avg_return_last10"], color="tab:blue", label="Avg Return (last 10 eps)")
    smooth_ret = moving_average(log_df["avg_return_last10"], window=5)
    axes[0, 0].plot(epochs, smooth_ret, color="tab:orange", label="Smoothed")
    axes[0, 0].set_title(f"RL Return ({mode})")
    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Return")
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    axes[0, 1].cla()
    axes[0, 1].plot(epochs, log_df["catch_rate_last10"], color="tab:red", label="Catch Rate (last 10 eps)")
    axes[0, 1].plot(epochs, log_df["avg_min_distance_last10"], color="tab:green", label="Avg Min Dist (last 10 eps)")
    axes[0, 1].set_title("Safety Metrics")
    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Value")
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()

    axes[1, 0].cla()
    axes[1, 0].plot(epochs, log_df["loss_total"], label="Total Loss", color="tab:blue")
    axes[1, 0].plot(epochs, log_df["policy_loss"], label="Policy Loss", color="tab:orange")
    axes[1, 0].plot(epochs, log_df["value_loss"], label="Value Loss", color="tab:green")
    axes[1, 0].set_title("PPO Loss Breakdown")
    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Loss")
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    axes[1, 1].cla()
    axes[1, 1].plot(epochs, log_df["entropy"], label="Entropy", color="tab:purple")
    axes[1, 1].plot(epochs, log_df["approx_kl"], label="Approx KL", color="tab:brown")
    axes[1, 1].plot(epochs, log_df["clip_frac"], label="Clip Fraction", color="tab:gray")
    axes[1, 1].set_title("Policy Health")
    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Value")
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()

    fig.tight_layout()
    fig.canvas.draw_idle()
    fig.canvas.flush_events()
    plt.pause(0.001)


def plot_rl_training_results(log_df, output_dir, mode):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    _update_rl_live_plot(fig, axes, log_df, mode)

    save_path = os.path.join(output_dir, f"rl_training_plot_{mode}.png")
    fig.savefig(save_path)
    print(f"RL training plot saved to {save_path}")
    plt.close(fig)


# --- Step 1: Collect Demonstrations ---

def collect_demonstrations(mode, num_episodes, output_dir, cfg):
    print(f"Collecting {num_episodes} Expert demos in {mode} mode...")
    env = DroneEnv(mode, config=cfg)
    expert = BlueEvasivePolicy(cfg)

    data = []

    for ep in range(num_episodes):
        obs = env.reset()
        done = False

        while not done:
            action = expert.get_action(obs, "blue")
            state_vec = env.get_flat_state(obs)

            record = {
                "episode_id": ep,
                "mode": mode,
                "state": state_vec,
                "action": action,
                "blue_pos": obs["blue"][0:2].copy(),
                "red_pos": obs["red"][0:2].copy(),
            }
            data.append(record)

            obs, _, done, _ = env.step(action)

    print(f"Collected {len(data)} transitions.")

    states = np.stack([d["state"] for d in data])
    if mode == "DISCRETE":
        actions = np.array([d["action"] for d in data])
    else:
        actions = np.stack([d["action"] for d in data])

    save_path = os.path.join(output_dir, f"expert_{mode}")
    if torch:
        torch.save({"states": states, "actions": actions}, save_path + ".pt")
        print(f"Expert data saved to {save_path}.pt")
    else:
        np.savez(save_path + ".npz", states=states, actions=actions)
        print(f"Expert data saved to {save_path}.npz (NumPy format)")

    return states, actions


# --- Step 2: Behavior Cloning ---

def train_bc(
    mode,
    states,
    actions,
    output_dir,
    cfg,
    epochs=50,
    batch_size=64,
    val_ratio=0.1,
    live_plot=False,
):
    if torch is None:
        raise RuntimeError("PyTorch is required for behavior cloning.")

    print(f"Training BC Policy for {mode}...")

    states = _ensure_numpy(states).astype(np.float32)
    actions = _ensure_numpy(actions)

    state_dim = states.shape[1]
    num_samples = states.shape[0]

    if mode == "DISCRETE":
        action_dim = 9
        criterion = nn.CrossEntropyLoss()
        actions_tensor = torch.tensor(actions, dtype=torch.long)
    else:
        action_dim = 2
        criterion = nn.MSELoss()
        actions_tensor = torch.tensor(actions, dtype=torch.float32)

    states_tensor = torch.tensor(states, dtype=torch.float32)

    agent = PPOAgent(state_dim, action_dim, mode)
    agent.model.to(DEVICE)
    optimizer = optim.Adam(agent.model.parameters(), lr=1e-3)

    if num_samples < 2:
        raise ValueError("Need at least 2 samples for BC training.")

    val_size = int(num_samples * val_ratio)
    val_size = min(max(1, val_size), num_samples - 1)

    indices = np.random.permutation(num_samples)
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_ds = TensorDataset(states_tensor[train_idx], actions_tensor[train_idx])
    val_ds = TensorDataset(states_tensor[val_idx], actions_tensor[val_idx])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    training_logs = []
    best_val_loss = float("inf")
    best_epoch = 1
    best_state_dict = None

    bc_live_fig = None
    bc_live_axes = None
    if live_plot:
        plt.ion()
        bc_live_fig, bc_live_axes = plt.subplots(1, 2, figsize=(12, 4.8))

    for epoch in range(epochs):
        agent.model.train()
        train_loss_sum = 0.0
        train_metric_sum = 0.0
        train_count = 0

        for bx, by in train_loader:
            bx = bx.to(DEVICE)
            by = by.to(DEVICE)

            optimizer.zero_grad()
            loss, metric = _bc_batch_metrics(agent, bx, by, mode, criterion, cfg)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.model.parameters(), 1.0)
            optimizer.step()

            batch_size_curr = bx.size(0)
            train_loss_sum += loss.item() * batch_size_curr
            train_metric_sum += metric.item() * batch_size_curr
            train_count += batch_size_curr

        train_loss = train_loss_sum / max(1, train_count)
        train_metric = train_metric_sum / max(1, train_count)

        agent.model.eval()
        val_loss_sum = 0.0
        val_metric_sum = 0.0
        val_count = 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx = bx.to(DEVICE)
                by = by.to(DEVICE)

                loss, metric = _bc_batch_metrics(agent, bx, by, mode, criterion, cfg)
                batch_size_curr = bx.size(0)
                val_loss_sum += loss.item() * batch_size_curr
                val_metric_sum += metric.item() * batch_size_curr
                val_count += batch_size_curr

        val_loss = val_loss_sum / max(1, val_count)
        val_metric = val_metric_sum / max(1, val_count)

        log_row = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        if mode == "DISCRETE":
            log_row["train_accuracy"] = train_metric
            log_row["val_accuracy"] = val_metric
            print(
                f"BC Epoch {epoch + 1}/{epochs} | train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | val_acc={val_metric:.3f}"
            )
        else:
            log_row["train_action_mae"] = train_metric
            log_row["val_action_mae"] = val_metric
            print(
                f"BC Epoch {epoch + 1}/{epochs} | train_loss={train_loss:.4f} | "
                f"val_loss={val_loss:.4f} | val_mae={val_metric:.4f}"
            )

        training_logs.append(log_row)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_state_dict = copy.deepcopy(agent.model.state_dict())

        if live_plot:
            _update_bc_live_plot(bc_live_fig, bc_live_axes, pd.DataFrame(training_logs), mode)

    if best_state_dict is not None:
        agent.model.load_state_dict(best_state_dict)

    save_path = os.path.join(output_dir, f"bc_model_{mode}.pth")
    torch.save(agent.model.state_dict(), save_path)
    print(f"BC Model saved to {save_path} (best epoch: {best_epoch})")

    log_df = pd.DataFrame(training_logs)
    bc_log_path = os.path.join(output_dir, f"bc_training_log_{mode}.csv")
    log_df.to_csv(bc_log_path, index=False)
    print(f"BC training logs saved to {bc_log_path}")

    plot_bc_training_results(log_df, output_dir, mode)

    if live_plot:
        plt.ioff()
        plt.close(bc_live_fig)

    return agent


# --- Step 3: RL Fine-tuning ---

def train_rl(
    mode,
    cfg,
    start_agent=None,
    timesteps=100000,
    output_dir="drone_data",
    steps_per_epoch=2048,
    live_plot=False,
):
    if torch is None:
        raise RuntimeError("PyTorch is required for RL fine-tuning.")

    print(f"Fine-tuning with PPO ({mode})...")

    env = DroneEnv(mode, config=cfg)
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()

    if start_agent is not None:
        agent = start_agent
    else:
        bc_path = os.path.join(output_dir, f"bc_model_{mode}.pth")
        agent = PPOAgent(state_dim, action_dim, mode)
        if os.path.exists(bc_path):
            print("Loading BC weights for initialization...")
            agent.model.load_state_dict(torch.load(bc_path, map_location=DEVICE))

    agent.model.to(DEVICE)
    agent.model.train()

    epochs = max(1, int(np.ceil(timesteps / steps_per_epoch)))

    obs = env.reset()
    curr_ep_ret = 0.0
    curr_ep_min_dist = float("inf")
    total_steps = 0

    ep_returns = []
    ep_min_dists = []
    ep_caught = []
    training_logs = []

    rl_live_fig = None
    rl_live_axes = None
    if live_plot:
        plt.ion()
        rl_live_fig, rl_live_axes = plt.subplots(2, 2, figsize=(13, 9))

    for epoch in range(epochs):
        if total_steps >= timesteps:
            break

        agent.clear_memory()
        steps_this_epoch = min(steps_per_epoch, timesteps - total_steps)
        finished_eps_this_epoch = 0

        for _ in range(steps_this_epoch):
            state_vec = env.get_flat_state(obs)
            s_tensor = torch.tensor(state_vec, dtype=torch.float32, device=DEVICE)

            with torch.no_grad():
                action, log_prob, val = agent.model.get_action(s_tensor)

            if mode == "DISCRETE":
                act_env = action.item()
                act_store = action.item()
            else:
                act_np = action.detach().cpu().numpy()
                act_env = act_np * cfg.BLUE_MAX_ACCEL
                act_store = act_np

            obs, reward, done, info = env.step(act_env)

            agent.store(
                state_vec,
                act_store,
                reward,
                val.item(),
                log_prob.item(),
                done,
            )

            curr_ep_ret += reward
            curr_ep_min_dist = min(curr_ep_min_dist, info.get("distance", env.get_distance()))
            total_steps += 1

            if done:
                ep_returns.append(curr_ep_ret)
                ep_min_dists.append(curr_ep_min_dist)
                ep_caught.append(1 if info["outcome"] == "caught" else 0)
                finished_eps_this_epoch += 1

                obs = env.reset()
                curr_ep_ret = 0.0
                curr_ep_min_dist = float("inf")

        last_state = env.get_flat_state(obs)
        ls_tensor = torch.tensor(last_state, dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            _, last_val = agent.model(ls_tensor)

        update_metrics = agent.update(last_val.item())

        avg_ret_last10 = float(np.mean(ep_returns[-10:])) if ep_returns else 0.0
        avg_dist_last10 = float(np.mean(ep_min_dists[-10:])) if ep_min_dists else 0.0
        catch_rate_last10 = float(np.mean(ep_caught[-10:])) if ep_caught else 0.0

        row = {
            "epoch": epoch + 1,
            "total_steps": total_steps,
            "episodes_completed_total": len(ep_returns),
            "episodes_completed_epoch": finished_eps_this_epoch,
            "avg_return_last10": avg_ret_last10,
            "avg_min_distance_last10": avg_dist_last10,
            "catch_rate_last10": catch_rate_last10,
        }
        row.update(update_metrics)
        training_logs.append(row)

        print(
            f"RL Epoch {epoch + 1}/{epochs} | steps={total_steps} | "
            f"avg_ret10={avg_ret_last10:.2f} | catch10={catch_rate_last10:.2f} | "
            f"loss={row['loss_total']:.4f} | kl={row['approx_kl']:.5f}"
        )

        if live_plot:
            _update_rl_live_plot(rl_live_fig, rl_live_axes, pd.DataFrame(training_logs), mode)

    save_path = os.path.join(output_dir, f"rl_model_{mode}.pth")
    torch.save(agent.model.state_dict(), save_path)
    print(f"RL Model saved to {save_path}")

    log_df = pd.DataFrame(training_logs)
    log_path = os.path.join(output_dir, f"training_log_{mode}.csv")
    log_df.to_csv(log_path, index=False)
    print(f"RL training logs saved to {log_path}")

    plot_rl_training_results(log_df, output_dir, mode)

    if live_plot:
        plt.ioff()
        plt.close(rl_live_fig)

    return agent, ep_returns


# --- Step 4: Full Evaluation ---

def evaluate_agents(mode, output_dir, cfg, visualize=False):
    print(f"\nEvaluating Agents ({mode} Mode)...")
    env = DroneEnv(mode, config=cfg)
    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()

    agents = {"Expert": BlueEvasivePolicy(cfg)}

    bc_agent = PPOAgent(state_dim, action_dim, mode)
    bc_path = os.path.join(output_dir, f"bc_model_{mode}.pth")
    if os.path.exists(bc_path):
        bc_agent.model.load_state_dict(torch.load(bc_path, map_location=DEVICE))
        bc_agent.model.to(DEVICE)
        bc_agent.model.eval()
        agents["BC"] = bc_agent

    rl_agent = PPOAgent(state_dim, action_dim, mode)
    rl_path = os.path.join(output_dir, f"rl_model_{mode}.pth")
    if os.path.exists(rl_path):
        rl_agent.model.load_state_dict(torch.load(rl_path, map_location=DEVICE))
        rl_agent.model.to(DEVICE)
        rl_agent.model.eval()
        agents["BC+RL"] = rl_agent

    results = []

    if visualize:
        plt.ion()
        _, ax = plt.subplots(figsize=(6, 6))

    for name, policy in agents.items():
        env.seed(cfg.SEED)
        print(f"Testing {name}...")
        for i in range(50):
            obs = env.reset()
            done = False
            trajectory = []

            while not done:
                if name == "Expert":
                    act_env = policy.get_action(obs, "blue")
                else:
                    state_vec = env.get_flat_state(obs)
                    s_tensor = torch.tensor(state_vec, dtype=torch.float32, device=DEVICE)
                    with torch.no_grad():
                        action, _, _ = policy.model.get_action(s_tensor, deterministic=True)

                    if mode == "DISCRETE":
                        act_env = action.item()
                    else:
                        act_env = action.detach().cpu().numpy() * cfg.BLUE_MAX_ACCEL

                obs_prev = obs
                obs, _, done, info = env.step(act_env)

                if visualize and name == "BC+RL" and i < 5:
                    env.render(ax)
                    plt.pause(0.01)

                trajectory.append(
                    {
                        "run_type": name,
                        "episode_id": i,
                        "t": env.t,
                        "step": env.step_count,
                        "mode": mode,
                        "blue_x": obs_prev["blue"][0],
                        "blue_y": obs_prev["blue"][1],
                        "red_x": obs_prev["red"][0],
                        "red_y": obs_prev["red"][1],
                        "distance": env.get_distance(),
                        "caught": 1 if info["outcome"] == "caught" else 0,
                    }
                )

            results.extend(trajectory)

    df = pd.DataFrame(results)
    csv_path = os.path.join(output_dir, f"evaluation_results_{mode}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved evaluation results to {csv_path}")

    episode_stats = (
        df.groupby(["run_type", "episode_id"])
        .agg({"caught": "max", "step": "max", "distance": "min"})
        .reset_index()
    )

    summary = episode_stats.groupby("run_type").agg(
        {"caught": "mean", "step": "mean", "distance": "mean"}
    )

    print("\n--- Results Summary ---")
    print(summary)
    print("-----------------------")

    if visualize:
        plt.ioff()
        plt.close()


# --- Main CLI ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="CONTINUOUS", choices=["CONTINUOUS", "DISCRETE"])
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--config", type=str, default=None, help="Path to train config YAML")
    parser.add_argument("--collect_demos", action="store_true", help="Step 1: Generate Expert Data")
    parser.add_argument("--train_bc", action="store_true", help="Step 2: Train Behavior Cloning")
    parser.add_argument("--train_rl", action="store_true", help="Step 3: Fine-tune with RL")
    parser.add_argument("--eval", action="store_true", help="Step 4: Evaluate all agents")
    parser.add_argument("--all", action="store_true", help="Run all steps in order")
    parser.add_argument("--episodes_demo", type=int, default=2000)
    parser.add_argument("--bc_epochs", type=int, default=30)
    parser.add_argument("--steps_rl", type=int, default=100000)
    parser.add_argument("--steps_per_epoch", type=int, default=2048)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--visualize", action="store_true", help="Visualize RL evaluation")
    parser.add_argument("--live_plots", action="store_true", help="Show live training plots during BC and RL")

    args = parser.parse_args()

    overrides = {}
    if args.seed is not None:
        overrides["SEED"] = args.seed
    if args.output_dir is not None:
        overrides["OUTPUT_DIR"] = args.output_dir
    cfg = load_env_config(profile="train", config_path=args.config, overrides=overrides)
    output_dir = cfg.OUTPUT_DIR

    os.makedirs(output_dir, exist_ok=True)
    set_global_seed(cfg.SEED)

    if args.all:
        args.collect_demos = True
        args.train_bc = True
        args.train_rl = True
        args.eval = True

    if torch is None and (args.train_bc or args.train_rl or args.eval):
        raise RuntimeError("PyTorch is required for --train_bc, --train_rl, and --eval.")

    if args.collect_demos:
        states, actions = collect_demonstrations(args.mode, args.episodes_demo, output_dir, cfg)
    elif args.train_bc:
        pt_path = os.path.join(output_dir, f"expert_{args.mode}.pt")
        if os.path.exists(pt_path):
            data = torch.load(pt_path, map_location="cpu")
            states, actions = data["states"], data["actions"]
        else:
            raise FileNotFoundError("Expert data not found. Run with --collect_demos first.")

    bc_agent = None
    if args.train_bc:
        bc_agent = train_bc(
            args.mode,
            states,
            actions,
            output_dir,
            cfg,
            epochs=args.bc_epochs,
            live_plot=args.live_plots,
        )

    if args.train_rl:
        train_rl(
            args.mode,
            cfg,
            start_agent=bc_agent,
            timesteps=args.steps_rl,
            output_dir=output_dir,
            steps_per_epoch=args.steps_per_epoch,
            live_plot=args.live_plots,
        )

    if args.eval:
        evaluate_agents(args.mode, output_dir, cfg, args.visualize)
