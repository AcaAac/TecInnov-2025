import argparse
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import build_episode_summary
from env import DroneEnv, RedPursuitPolicy, load_env_config

try:
    import torch

    if torch.backends.mps.is_available():
        DEVICE = torch.device("mps")
    else:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    torch = None
    DEVICE = None

from ppo import PPOAgent


def _extract_positions(env, obs):
    if env.mode == "DISCRETE":
        b_pos = env._idx_to_pos(obs["blue"])
        r_pos = env._idx_to_pos(obs["red"])
        b_vel = np.zeros(2, dtype=float)
        r_vel = np.zeros(2, dtype=float)
    else:
        b_pos = np.asarray(obs["blue"][0:2], dtype=float)
        r_pos = np.asarray(obs["red"][0:2], dtype=float)
        b_vel = np.asarray(obs["blue"][2:4], dtype=float)
        r_vel = np.asarray(obs["red"][2:4], dtype=float)
    return b_pos, r_pos, b_vel, r_vel


def _resolve_model_path(output_dir: str, mode: str, model_path: Optional[str]) -> str:
    if model_path:
        path = os.path.abspath(os.path.expanduser(model_path))
    else:
        path = os.path.abspath(os.path.join(output_dir, f"rl_model_{mode}.pth"))

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"RL model not found at '{path}'. Run training first, e.g.: "
            f"python src/train_blue.py --mode {mode} --config configs/train.yaml --train_rl"
        )
    return path


def _run_rl_episode(
    env,
    rl_agent,
    red_policy,
    mode: str,
    deterministic: bool,
    episode_id: int,
    visualize: bool,
    ax,
    render_dt: float,
):
    obs = env.reset()
    done = False
    trajectory = []

    init_b_pos, init_r_pos, _, _ = _extract_positions(env, obs)
    init_distance = env.get_distance()

    while not done:
        state_vec = env.get_flat_state(obs)
        s_tensor = torch.tensor(state_vec, dtype=torch.float32, device=DEVICE)

        with torch.no_grad():
            action, _, _ = rl_agent.model.get_action(s_tensor, deterministic=deterministic)

        if mode == "DISCRETE":
            act_blue = action.item()
        else:
            act_blue = action.detach().cpu().numpy() * env.config.V_BLUE_MAX

        act_red = red_policy.get_action(obs, "red")
        obs, _, done, info = env.step(act_blue, act_red)

        b_pos, r_pos, b_vel, r_vel = _extract_positions(env, obs)
        distance = float(info.get("distance", env.get_distance()))
        captured = int(bool(info.get("caught", False)))
        outcome = str(info.get("outcome", "running"))

        trajectory.append(
            {
                "framework": "test_rl",
                "policy_name": "RL",
                "episode_id": int(episode_id),
                "step": int(env.step_count),
                "time": float(env.t),
                "mode": mode,
                "blue_x": float(b_pos[0]),
                "blue_y": float(b_pos[1]),
                "red_x": float(r_pos[0]),
                "red_y": float(r_pos[1]),
                "blue_vx": float(b_vel[0]),
                "blue_vy": float(b_vel[1]),
                "red_vx": float(r_vel[0]),
                "red_vy": float(r_vel[1]),
                "distance": distance,
                "captured": captured,
                "outcome": outcome,
                "init_blue_x": float(init_b_pos[0]),
                "init_blue_y": float(init_b_pos[1]),
                "init_red_x": float(init_r_pos[0]),
                "init_red_y": float(init_r_pos[1]),
                "init_distance": float(init_distance),
            }
        )

        if visualize:
            env.render(ax)
            plt.pause(render_dt)

    return trajectory


def main():
    if torch is None:
        raise RuntimeError("PyTorch is required to run RL model visualization.")

    parser = argparse.ArgumentParser(description="Visualize trained RL Blue policy against Red pursuer")
    parser.add_argument("--mode", type=str, default="CONTINUOUS", choices=["CONTINUOUS", "DISCRETE"])
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML (default: train profile)")
    parser.add_argument("--model_path", type=str, default=None, help="Path to rl_model_*.pth")
    parser.add_argument("--seed", type=int, default=None, help="Override seed")
    parser.add_argument("--output_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--no_visualize", action="store_true", help="Disable live visualization window")
    parser.add_argument("--stochastic", action="store_true", help="Sample stochastic actions (default deterministic)")
    parser.add_argument("--render_dt", type=float, default=0.03, help="Seconds per rendered frame")
    parser.add_argument("--episode_pause", type=float, default=0.4, help="Pause between episodes (seconds)")
    parser.add_argument("--no_hold", action="store_true", help="Close plot at end instead of keeping it open")

    args = parser.parse_args()

    overrides = {}
    if args.seed is not None:
        overrides["SEED"] = args.seed
    if args.output_dir is not None:
        overrides["OUTPUT_DIR"] = args.output_dir

    cfg = load_env_config(profile="train", config_path=args.config, overrides=overrides)
    model_path = _resolve_model_path(cfg.OUTPUT_DIR, args.mode, args.model_path)

    env = DroneEnv(mode=args.mode, config=cfg)
    red_policy = RedPursuitPolicy(cfg)

    state_dim = env.get_state_dim()
    action_dim = env.get_action_dim()

    rl_agent = PPOAgent(state_dim, action_dim, args.mode)
    rl_agent.model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    rl_agent.model.to(DEVICE)
    rl_agent.model.eval()

    visualize = not args.no_visualize
    all_steps = []

    if visualize:
        plt.ion()
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = None
        ax = None

    print(f"Loaded RL model: {model_path}")
    print(f"Running {args.episodes} episodes in {args.mode} mode...")

    for ep in range(args.episodes):
        env.seed(cfg.SEED + ep)
        episode_steps = _run_rl_episode(
            env=env,
            rl_agent=rl_agent,
            red_policy=red_policy,
            mode=args.mode,
            deterministic=not args.stochastic,
            episode_id=ep,
            visualize=visualize,
            ax=ax,
            render_dt=args.render_dt,
        )
        all_steps.extend(episode_steps)

        ep_df = pd.DataFrame(episode_steps)
        captured = int(ep_df["captured"].max()) if not ep_df.empty else 0
        outcome = "caught" if captured else "timeout"
        min_dist = float(ep_df["distance"].min()) if not ep_df.empty else float("nan")
        print(f"Episode {ep + 1}/{args.episodes}: outcome={outcome}, steps={len(ep_df)}, min_dist={min_dist:.4f}")
        if visualize and ep < args.episodes - 1:
            plt.pause(max(0.0, args.episode_pause))

    if visualize:
        plt.ioff()

    step_df = pd.DataFrame(all_steps)
    episode_df = build_episode_summary(step_df, group_cols=("policy_name", "episode_id"))

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    step_csv = os.path.join(cfg.OUTPUT_DIR, f"test_rl_steps_{args.mode}.csv")
    ep_csv = os.path.join(cfg.OUTPUT_DIR, f"test_rl_episodes_{args.mode}.csv")
    step_df.to_csv(step_csv, index=False)
    episode_df.to_csv(ep_csv, index=False)

    capture_rate = float(episode_df["captured"].mean()) if not episode_df.empty else float("nan")
    avg_steps = float(episode_df["steps_to_terminal"].mean()) if not episode_df.empty else float("nan")
    avg_min_dist = float(episode_df["min_distance"].mean()) if not episode_df.empty else float("nan")

    print("-" * 40)
    print("RL Test Summary")
    print(f"Capture Rate: {capture_rate * 100:.1f}%")
    print(f"Avg Steps: {avg_steps:.1f}")
    print(f"Avg Min Dist: {avg_min_dist:.4f}")
    print(f"Saved: {step_csv}")
    print(f"Saved: {ep_csv}")
    print("-" * 40)

    if visualize:
        if args.no_hold:
            plt.close(fig)
        else:
            print("Visualization finished. Close the plot window to exit.")
            plt.show(block=True)


if __name__ == "__main__":
    main()
