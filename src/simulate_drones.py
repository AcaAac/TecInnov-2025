import argparse
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from env import BlueEvasivePolicy, DroneEnv, RedPursuitPolicy, load_env_config


def run_episode(env, blue_policy, red_policy, episode_id, render_callback=None):
    obs = env.reset()
    done = False
    trajectory = []

    while not done:
        act_blue = blue_policy.get_action(obs, "blue")
        act_red = red_policy.get_action(obs, "red")

        obs_curr = obs
        obs, _, done, info = env.step(act_blue, act_red)
        caught = bool(info.get("caught", False))

        if env.mode == "DISCRETE":
            b_pos = env._idx_to_pos(obs_curr["blue"])
            r_pos = env._idx_to_pos(obs_curr["red"])
            b_vel = [0.0, 0.0]
            r_vel = [0.0, 0.0]
        else:
            b_pos = obs_curr["blue"][0:2]
            r_pos = obs_curr["red"][0:2]
            b_vel = obs_curr["blue"][2:4]
            r_vel = obs_curr["red"][2:4]

        dist = float(np.linalg.norm(b_pos - r_pos))

        step_data = {
            "episode_id": episode_id,
            "step": env.step_count,
            "time": env.t,
            "mode": env.mode,
            "blue_x": float(b_pos[0]),
            "blue_y": float(b_pos[1]),
            "red_x": float(r_pos[0]),
            "red_y": float(r_pos[1]),
            "blue_vx": float(b_vel[0]),
            "blue_vy": float(b_vel[1]),
            "red_vx": float(r_vel[0]),
            "red_vy": float(r_vel[1]),
            "distance": dist,
            "caught": caught,
        }
        trajectory.append(step_data)

        if render_callback:
            render_callback(b_pos, r_pos)

    return trajectory, caught


def run_batch_simulation(num_episodes, mode, cfg, show_anim=False):
    print(f"Starting {num_episodes} episodes in {mode} mode...")

    env = DroneEnv(mode=mode, config=cfg)
    blue_pol = BlueEvasivePolicy(cfg, seed=cfg.SEED)
    red_pol = RedPursuitPolicy(cfg)

    all_data = []

    if show_anim:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(0, cfg.ARENA_SIZE)
        ax.set_ylim(0, cfg.ARENA_SIZE)
        ax.set_title(f"Drone Pursuit ({mode})")
        blue_dot, = ax.plot([], [], "bo", markersize=10, label="Blue (Evader)")
        red_dot, = ax.plot([], [], "ro", markersize=10, label="Red (Pursuer)")

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
            blue_dot.set_data([b[0]], [b[1]])
            red_dot.set_data([r[0]], [r[1]])
            capture_circle.center = (r[0], r[1])
            plt.pause(0.001)

    else:
        update_render = None

    start_time = time.time()

    stats_caught = 0
    stats_lens = []
    stats_min_dists = []

    for i in range(num_episodes):
        env.seed(cfg.SEED + i)

        traj, caught = run_episode(env, blue_pol, red_pol, i, update_render)
        all_data.extend(traj)

        stats_caught += int(caught)
        stats_lens.append(len(traj))
        dists = [d["distance"] for d in traj]
        stats_min_dists.append(min(dists))

        if (i + 1) % 50 == 0:
            print(f"Completed {i + 1}/{num_episodes}...")

    if show_anim:
        plt.ioff()
        plt.close()

    duration = time.time() - start_time
    print(f"Simulation finished in {duration:.2f}s")

    print("-" * 30)
    print(f"Summary ({mode}):")
    print(f"Catch Rate: {stats_caught / num_episodes * 100:.1f}%")
    print(f"Avg Steps: {np.mean(stats_lens):.1f}")
    print(f"Avg Min Dist: {np.mean(stats_min_dists):.4f}")
    print("-" * 30)

    return pd.DataFrame(all_data), stats_lens, stats_min_dists


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

    df, lens, min_dists = run_batch_simulation(episodes, args.mode, cfg, args.visualize)

    csv_path = os.path.join(cfg.OUTPUT_DIR, "drone_dataset.csv")
    df.to_csv(csv_path, index=False)

    try:
        import pyarrow  # noqa: F401

        pq_path = os.path.join(cfg.OUTPUT_DIR, "drone_dataset.parquet")
        df.to_parquet(pq_path, index=False)
        pq_msg = f"and {pq_path}"
    except ImportError:
        pq_msg = "(PyArrow not found, skipping Parquet)"

    print(f"Dataset saved to: {csv_path} {pq_msg}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.hist(lens, bins=20, color="skyblue", edgecolor="black")
    ax1.set_title("Episode Lengths (Steps)")
    ax1.set_xlabel("Steps")

    ax2.hist(min_dists, bins=20, color="salmon", edgecolor="black")
    ax2.set_title("Min Distances Achieved")
    ax2.set_xlabel("Distance")
    ax2.vlines(cfg.CAPTURE_RADIUS, 0, ax2.get_ylim()[1], colors="r", linestyles="dashed", label="Catch Radius")
    ax2.legend()

    plt.tight_layout()
    plot_path = os.path.join(cfg.OUTPUT_DIR, "simulation_summary.png")
    plt.savefig(plot_path)
    print(f"Summary plot saved to: {plot_path}")
