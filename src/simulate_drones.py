import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import (
    build_episode_summary,
    plot_success_vs_initial_distance,
    plot_time_to_capture_by_policy,
)
from env import EvaderPolicy, DroneEnv, PursuerPolicy, load_env_config


def _extract_positions(env, obs):
    if env.mode == "DISCRETE":
        evader_pos = env._idx_to_pos(obs["evader"])
        pursuer_pos = env._idx_to_pos(obs["pursuer"])
        evader_vel = np.zeros(2, dtype=float)
        pursuer_vel = np.zeros(2, dtype=float)
    else:
        evader_pos = np.asarray(obs["evader"][0:2], dtype=float)
        pursuer_pos = np.asarray(obs["pursuer"][0:2], dtype=float)
        evader_vel = np.asarray(obs["evader"][2:4], dtype=float)
        pursuer_vel = np.asarray(obs["pursuer"][2:4], dtype=float)
    return evader_pos, pursuer_pos, evader_vel, pursuer_vel


def run_episode(
    env,
    evader_policy,
    pursuer_policy,
    episode_id,
    policy_name="HeuristicEvader",
    render_callback=None,
):
    obs = env.reset()
    done = False
    trajectory = []

    init_evader_pos, init_pursuer_pos, _, _ = _extract_positions(env, obs)
    init_distance = env.get_distance()

    while not done:
        act_evader = evader_policy.get_action(obs, "evader")
        act_pursuer = pursuer_policy.get_action(obs, "pursuer")

        obs, _, done, info = env.step(act_evader, act_pursuer)

        # Log post-step state so position, step/time, and info metrics are aligned.
        evader_pos, pursuer_pos, evader_vel, pursuer_vel = _extract_positions(env, obs)
        distance = float(info.get("distance", env.get_distance()))
        captured = int(bool(info.get("caught", False)))
        outcome = str(info.get("outcome", "running"))

        trajectory.append(
            {
                "framework": "simulate",
                "policy_name": policy_name,
                "episode_id": int(episode_id),
                "step": int(env.step_count),
                "time": float(env.t),
                "mode": env.mode,
                "evader_x": float(evader_pos[0]),
                "evader_y": float(evader_pos[1]),
                "pursuer_x": float(pursuer_pos[0]),
                "pursuer_y": float(pursuer_pos[1]),
                "evader_vx": float(evader_vel[0]),
                "evader_vy": float(evader_vel[1]),
                "pursuer_vx": float(pursuer_vel[0]),
                "pursuer_vy": float(pursuer_vel[1]),
                "distance": distance,
                "captured": captured,
                "outcome": outcome,
                "init_evader_x": float(init_evader_pos[0]),
                "init_evader_y": float(init_evader_pos[1]),
                "init_pursuer_x": float(init_pursuer_pos[0]),
                "init_pursuer_y": float(init_pursuer_pos[1]),
                "init_distance": float(init_distance),
            }
        )

        if render_callback:
            render_callback(evader_pos, pursuer_pos)

    return trajectory


def plot_simulation_summary(episode_df, cfg, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    if episode_df.empty:
        for ax in axes.ravel():
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output_path)
        plt.close(fig)
        return

    axes[0, 0].hist(episode_df["steps_to_terminal"], bins=20, color="skyblue", edgecolor="black")
    axes[0, 0].set_title("Episode Lengths (steps)")
    axes[0, 0].set_xlabel("Steps")
    axes[0, 0].set_ylabel("Count")

    axes[0, 1].hist(episode_df["min_distance"], bins=20, color="salmon", edgecolor="black")
    axes[0, 1].axvline(cfg.CAPTURE_RADIUS, color="red", linestyle="--", label="Capture Radius")
    axes[0, 1].set_title("Minimum Distance Distribution")
    axes[0, 1].set_xlabel("Distance")
    axes[0, 1].legend()

    plot_time_to_capture_by_policy(episode_df, ax=axes[1, 0], title="Time-to-Capture")
    plot_success_vs_initial_distance(
        episode_df,
        ax=axes[1, 1],
        n_bins=10,
        title="Success vs Initial Distance",
    )

    fig.suptitle("Simulation Evaluation Summary", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path)
    plt.close(fig)


def run_batch_simulation(num_episodes, mode, cfg, show_anim=False):
    print(f"Starting {num_episodes} episodes in {mode} mode...")

    env = DroneEnv(mode=mode, config=cfg)
    evader_pol = EvaderPolicy(cfg, seed=cfg.SEED)
    pursuer_pol = PursuerPolicy(cfg)

    all_steps = []

    if show_anim:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(0, cfg.ARENA_SIZE)
        ax.set_ylim(0, cfg.ARENA_SIZE)
        ax.set_title(f"Drone Pursuit ({mode})")
        evader_dot, = ax.plot([], [], "bo", markersize=10, label="Evader")
        pursuer_dot, = ax.plot([], [], "ro", markersize=10, label="Pursuer")

        capture_circle = plt.Circle(
            (0, 0),
            cfg.CAPTURE_RADIUS,
            color="red",
            fill=False,
            linestyle="--",
            linewidth=2,
            alpha=0.5,
            label="Capture Radius",
        )
        ax.add_patch(capture_circle)

        ax.legend()
        ax.set_aspect("equal")
        plt.ion()
        plt.show()

        def update_render(b, r):
            evader_dot.set_data([b[0]], [b[1]])
            pursuer_dot.set_data([r[0]], [r[1]])
            capture_circle.center = (r[0], r[1])
            plt.pause(0.001)

    else:
        update_render = None

    start_time = time.time()

    for i in range(num_episodes):
        env.seed(cfg.SEED + i)
        steps = run_episode(env, evader_pol, pursuer_pol, i, render_callback=update_render)
        all_steps.extend(steps)

        if (i + 1) % 50 == 0:
            print(f"Completed {i + 1}/{num_episodes}...")

    if show_anim:
        plt.ioff()
        plt.close()

    step_df = pd.DataFrame(all_steps)
    episode_df = build_episode_summary(step_df, group_cols=("policy_name", "episode_id"))

    duration = time.time() - start_time
    print(f"Simulation finished in {duration:.2f}s")

    capture_rate = float(episode_df["captured"].mean()) if not episode_df.empty else float("nan")
    avg_steps = float(episode_df["steps_to_terminal"].mean()) if not episode_df.empty else float("nan")
    avg_min_dist = float(episode_df["min_distance"].mean()) if not episode_df.empty else float("nan")

    print("-" * 30)
    print(f"Summary ({mode}):")
    print(f"Capture Rate: {capture_rate * 100:.1f}%")
    print(f"Avg Steps: {avg_steps:.1f}")
    print(f"Avg Min Dist: {avg_min_dist:.4f}")
    print("-" * 30)

    return step_df, episode_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drone-on-Drone Simulation")
    parser.add_argument("--mode", type=str, default="CONTINUOUS", choices=["CONTINUOUS", "DISCRETE"], help="Simulation mode")
    parser.add_argument("--visualize", action="store_true", help="Show live animation (slow)")
    parser.add_argument("--episodes", type=int, default=None, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed")
    parser.add_argument("--output_dir", type=str, default=None, help="Override config output directory")
    parser.add_argument("--config", type=str, default=None, help="Path to simulate config YAML")

    args = parser.parse_args()

    overrides = {}
    if args.seed is not None:
        overrides["SEED"] = args.seed
    if args.output_dir is not None:
        overrides["OUTPUT_DIR"] = args.output_dir

    cfg = load_env_config(profile="simulate", config_path=args.config, overrides=overrides)
    episodes = cfg.NUM_EPISODES if args.episodes is None else args.episodes

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    step_df, episode_df = run_batch_simulation(episodes, args.mode, cfg, args.visualize)

    step_csv_path = os.path.join(cfg.OUTPUT_DIR, "drone_dataset.csv")
    step_df.to_csv(step_csv_path, index=False)

    episodes_csv_path = os.path.join(cfg.OUTPUT_DIR, "drone_dataset_episodes.csv")
    episode_df.to_csv(episodes_csv_path, index=False)

    try:
        import pyarrow  # noqa: F401

        step_pq_path = os.path.join(cfg.OUTPUT_DIR, "drone_dataset.parquet")
        step_df.to_parquet(step_pq_path, index=False)

        episodes_pq_path = os.path.join(cfg.OUTPUT_DIR, "drone_dataset_episodes.parquet")
        episode_df.to_parquet(episodes_pq_path, index=False)
        pq_msg = f"and {step_pq_path}, {episodes_pq_path}"
    except ImportError:
        pq_msg = "(PyArrow not found, skipping Parquet)"

    print(f"Dataset saved to: {step_csv_path}, {episodes_csv_path} {pq_msg}")

    plot_path = os.path.join(cfg.OUTPUT_DIR, "simulation_summary.png")
    plot_simulation_summary(episode_df, cfg, plot_path)
    print(f"Summary plot saved to: {plot_path}")
