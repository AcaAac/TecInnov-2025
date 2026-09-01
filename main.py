import argparse
import json
from pathlib import Path

from models.PPO import PPOTrainer
from utils import save_metrics_and_plots, setup_output_dirs


def _merge_config(base: dict[str, object], overrides: dict[str, object]) -> dict[str, object]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge_config(merged[key], value)  # type: ignore[arg-type]
        else:
            merged[key] = value
    return merged


def load_env_config(path: Path | None, seen: set[Path] | None = None) -> dict[str, object]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Environment config file not found: {path}")
    resolved_path = path.resolve()
    seen = set() if seen is None else seen
    if resolved_path in seen:
        raise ValueError(f"Circular environment config inheritance: {resolved_path}")
    seen.add(resolved_path)
    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    if not isinstance(config, dict):
        raise ValueError(f"Environment config must contain a JSON object: {path}")
    parent_name = config.pop("_extends", None)
    if parent_name is None:
        return config
    if not isinstance(parent_name, str):
        raise ValueError(f"_extends must be a string in environment config: {path}")
    parent_config = load_env_config(path.parent / parent_name, seen)
    return _merge_config(parent_config, config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate PPO on 3D leader-follower drone env.")
    parser.add_argument("--train-iters", type=int, default=200, help="Number of PPO training iterations.")
    parser.add_argument("--eval-episodes", type=int, default=10, help="Number of evaluation episodes.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Root directory for all artifacts.")
    parser.add_argument("--video", action="store_true", help="Save evaluation visualization videos.")
    # Use a slower default playback rate so rapid attitude corrections are
    # visually readable instead of appearing as exaggerated shaking.
    parser.add_argument("--video-fps", type=int, default=12, help="FPS for saved evaluation videos.")
    parser.add_argument(
        "--video-episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes to save as video.",
    )
    parser.add_argument("--env-config", type=str, default=None, help="Path to a JSON env config file.")
    parser.add_argument("--dt", type=float, default=None, help="Simulation timestep.")
    parser.add_argument("--max-steps", type=int, default=None, help="Max steps per episode.")
    parser.add_argument("--world-limit", type=float, default=None, help="World coordinate limit.")
    parser.add_argument("--desired-follow-distance", type=float, default=None, help="Desired follow distance.")
    parser.add_argument("--min-separation", type=float, default=None, help="Minimum separation.")
    parser.add_argument(
        "--more-realistic",
        action="store_true",
        help="Use opt-in AR3-like nonlinear 6-DOF dynamics for the blue drone.",
    )
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    dirs = setup_output_dirs(output_dir=output_dir, save_video=args.video)

    env_config = load_env_config(Path(args.env_config) if args.env_config else None)
    if args.dt is not None:
        env_config["dt"] = args.dt
    if args.max_steps is not None:
        env_config["max_steps"] = args.max_steps
    if args.world_limit is not None:
        env_config["world_limit"] = args.world_limit
    if args.desired_follow_distance is not None:
        env_config["desired_follow_distance"] = args.desired_follow_distance
    if args.min_separation is not None:
        env_config["min_separation"] = args.min_separation
    if args.more_realistic:
        env_config["more_realistic"] = True

    trainer = PPOTrainer(env_config=env_config)

    train_metrics = trainer.train(
        num_iterations=args.train_iters,
        episode_video_every=100,
        episode_video_dir=str(dirs["videos_dir"]),
        episode_video_fps=args.video_fps,
    )
    eval_metrics = trainer.evaluate(
        num_episodes=args.eval_episodes,
        deterministic=True,
        save_video=args.video,
        video_dir=str(dirs["videos_dir"]),
        video_fps=args.video_fps,
        video_episodes=args.video_episodes,
    )
    checkpoint_path = trainer.save(output_dir=str(dirs["checkpoints_dir"]))
    save_metrics_and_plots(output_dir=output_dir, train_metrics=train_metrics, eval_metrics=eval_metrics)

    print("\nTraining complete.")
    print(f"Last train iteration metrics: {train_metrics[-1] if train_metrics else {}}")
    print(
        "Evaluation summary: "
        f"escape_rate={eval_metrics['escape_rate']:.2%}, "
        f"mean_escape_time_s={eval_metrics['mean_escape_time_s']:.2f}, "
        f"mean_episode_reward={eval_metrics['mean_episode_reward']:.3f}"
    )
    print(f"Checkpoint saved at: {checkpoint_path}")
    print(f"Metrics saved under: {dirs['metrics_dir']}")
    print(f"Plots saved under: {dirs['plots_dir']}")
    print(f"Videos saved under: {dirs['videos_dir']}")


if __name__ == "__main__":
    main()
