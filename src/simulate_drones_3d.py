import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis import build_episode_summary_3d, plot_trajectory_projections_3d, plot_time_to_capture_by_policy_3d, plot_success_vs_initial_distance_3d
from analysis.eval_metrics_3d import plot_trajectory_scatter_3d
from env import BlueEvasivePolicy3D, DroneEnv3D, RedPursuitPolicy3D, load_env_config_3d


def _extract_positions(env, obs):
    b_pos = np.asarray(obs["blue"][0:3], dtype=float)
    r_pos = np.asarray(obs["red"][0:3], dtype=float)
    b_vel = np.asarray(obs["blue"][3:6], dtype=float)
    r_vel = np.asarray(obs["red"][3:6], dtype=float)
    return b_pos, r_pos, b_vel, r_vel


def run_episode(
    env,
    blue_policy,
    red_policy,
    episode_id,
    policy_name="HeuristicBlue3D",
    render_callback=None,
):
    obs = env.reset()
    done = False
    trajectory = []

    init_b_pos, init_r_pos, _, _ = _extract_positions(env, obs)
    init_distance = env.get_distance()

    while not done:
        act_blue = blue_policy.get_action(obs, "blue")
        act_red = red_policy.get_action(obs, "red")

        obs, _, done, info = env.step(act_blue, act_red)

        b_pos, r_pos, b_vel, r_vel = _extract_positions(env, obs)
        distance = float(info.get("distance", env.get_distance()))
        captured = int(bool(info.get("caught", False)))
        outcome = str(info.get("outcome", "running"))

        trajectory.append(
            {
                "framework": "simulate_3d",
                "policy_name": policy_name,
                "episode_id": int(episode_id),
                "step": int(env.step_count),
                "time": float(env.t),
                "mode": env.mode,
                "blue_x": float(b_pos[0]),
                "blue_y": float(b_pos[1]),
                "blue_z": float(b_pos[2]),
                "red_x": float(r_pos[0]),
                "red_y": float(r_pos[1]),
                "red_z": float(r_pos[2]),
                "blue_vx": float(b_vel[0]),
                "blue_vy": float(b_vel[1]),
                "blue_vz": float(b_vel[2]),
                "red_vx": float(r_vel[0]),
                "red_vy": float(r_vel[1]),
                "red_vz": float(r_vel[2]),
                "distance": distance,
                "captured": captured,
                "outcome": outcome,
                "init_blue_x": float(init_b_pos[0]),
                "init_blue_y": float(init_b_pos[1]),
                "init_blue_z": float(init_b_pos[2]),
                "init_red_x": float(init_r_pos[0]),
                "init_red_y": float(init_r_pos[1]),
                "init_red_z": float(init_r_pos[2]),
                "init_distance": float(init_distance),
            }
        )

        if render_callback:
            render_callback(b_pos, r_pos)

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

    plot_time_to_capture_by_policy_3d(episode_df, ax=axes[1, 0], title="Time-to-Capture")
    plot_success_vs_initial_distance_3d(
        episode_df,
        ax=axes[1, 1],
        n_bins=10,
        title="Success vs Initial Distance",
    )

    fig.suptitle("3D Simulation Evaluation Summary", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(output_path)
    plt.close(fig)


def _build_sample_trajectories(step_df, max_episodes=9):
    traj_map = {}
    for i, (episode_id, group) in enumerate(step_df.groupby("episode_id", sort=False)):
        if i >= max_episodes:
            break
        traj_map[(i, 0)] = group.sort_values("step")
    return traj_map


def run_batch_simulation(num_episodes, cfg, show_anim=False):
    print(f"Starting {num_episodes} episodes in 3D mode...")

    env = DroneEnv3D(config=cfg)
    blue_pol = BlueEvasivePolicy3D(cfg, seed=cfg.SEED)
    red_pol = RedPursuitPolicy3D(cfg)

    all_steps = []

    if show_anim:
        fig = plt.figure(figsize=(8, 7))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_xlim(0, cfg.ARENA_SIZE)
        ax.set_ylim(0, cfg.ARENA_SIZE)
        ax.set_zlim(0, cfg.ARENA_HEIGHT)
        ax.set_title("Drone Pursuit (3D)")
        plt.ion()
        plt.show()

        def update_render(b, r):
            ax.clear()
            ax.set_xlim(0, cfg.ARENA_SIZE)
            ax.set_ylim(0, cfg.ARENA_SIZE)
            ax.set_zlim(0, cfg.ARENA_HEIGHT)
            ax.scatter([b[0]], [b[1]], [b[2]], c="blue", s=40, label="Blue (Evader)")
            ax.scatter([r[0]], [r[1]], [r[2]], c="red", s=40, label="Red (Pursuer)")
            ax.plot([b[0], r[0]], [b[1], r[1]], [b[2], r[2]], color="gray", alpha=0.25)
            ax.legend(loc="upper right")
            plt.pause(0.001)

    else:
        update_render = None

    start_time = time.time()

    for i in range(num_episodes):
        env.seed(cfg.SEED + i)
        steps = run_episode(env, blue_pol, red_pol, i, render_callback=update_render)
        all_steps.extend(steps)

        if (i + 1) % 50 == 0:
            print(f"Completed {i + 1}/{num_episodes}...")

    if show_anim:
        plt.ioff()
        plt.close()

    step_df = pd.DataFrame(all_steps)
    episode_df = build_episode_summary_3d(step_df, group_cols=("policy_name", "episode_id"))

    duration = time.time() - start_time
    print(f"Simulation finished in {duration:.2f}s")

    capture_rate = float(episode_df["captured"].mean()) if not episode_df.empty else float("nan")
    avg_steps = float(episode_df["steps_to_terminal"].mean()) if not episode_df.empty else float("nan")
    avg_min_dist = float(episode_df["min_distance"].mean()) if not episode_df.empty else float("nan")

    print("-" * 30)
    print("Summary (3D):")
    print(f"Capture Rate: {capture_rate * 100:.1f}%")
    print(f"Avg Steps: {avg_steps:.1f}")
    print(f"Avg Min Dist: {avg_min_dist:.4f}")
    print("-" * 30)

    return step_df, episode_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Drone-on-Drone 3D Simulation")
    parser.add_argument("--visualize", action="store_true", help="Show live animation (slow)")
    parser.add_argument("--episodes", type=int, default=None, help="Number of episodes")
    parser.add_argument("--seed", type=int, default=None, help="Override config seed")
    parser.add_argument("--output_dir", type=str, default=None, help="Override config output directory")
    parser.add_argument("--config", type=str, default=None, help="Path to simulate 3D config YAML")

    args = parser.parse_args()

    overrides = {}
    if args.seed is not None:
        overrides["SEED"] = args.seed
    if args.output_dir is not None:
        overrides["OUTPUT_DIR"] = args.output_dir

    cfg = load_env_config_3d(profile="simulate", config_path=args.config, overrides=overrides)
    episodes = cfg.NUM_EPISODES if args.episodes is None else args.episodes

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    step_df, episode_df = run_batch_simulation(episodes, cfg, args.visualize)

    step_csv_path = os.path.join(cfg.OUTPUT_DIR, "drone_dataset_3D.csv")
    step_df.to_csv(step_csv_path, index=False)

    episodes_csv_path = os.path.join(cfg.OUTPUT_DIR, "drone_dataset_episodes_3D.csv")
    episode_df.to_csv(episodes_csv_path, index=False)

    try:
        import pyarrow  # noqa: F401

        step_pq_path = os.path.join(cfg.OUTPUT_DIR, "drone_dataset_3D.parquet")
        step_df.to_parquet(step_pq_path, index=False)

        episodes_pq_path = os.path.join(cfg.OUTPUT_DIR, "drone_dataset_episodes_3D.parquet")
        episode_df.to_parquet(episodes_pq_path, index=False)
        pq_msg = f"and {step_pq_path}, {episodes_pq_path}"
    except ImportError:
        pq_msg = "(PyArrow not found, skipping Parquet)"

    print(f"Dataset saved to: {step_csv_path}, {episodes_csv_path} {pq_msg}")

    plot_path = os.path.join(cfg.OUTPUT_DIR, "simulation_summary_3D.png")
    plot_simulation_summary(episode_df, cfg, plot_path)
    print(f"Summary plot saved to: {plot_path}")

    traj_map = _build_sample_trajectories(step_df, max_episodes=9)
    if traj_map:
        proj_fig, _ = plot_trajectory_projections_3d(traj_map, cfg.ARENA_SIZE, cfg.ARENA_HEIGHT)
        proj_path = os.path.join(cfg.OUTPUT_DIR, "simulation_trajectories_3D.png")
        proj_fig.savefig(proj_path)
        plt.close(proj_fig)
        print(f"Trajectory projection plot saved to: {proj_path}")

        scatter_ax = plot_trajectory_scatter_3d(traj_map, cfg.ARENA_SIZE, cfg.ARENA_HEIGHT)
        scatter_fig = scatter_ax.figure
        scatter_path = os.path.join(cfg.OUTPUT_DIR, "simulation_trajectories_3D_scatter.png")
        scatter_fig.savefig(scatter_path)
        plt.close(scatter_fig)
        print(f"3D scatter plot saved to: {scatter_path}")
