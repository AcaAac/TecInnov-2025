from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


def setup_output_dirs(output_dir: Path, save_video: bool) -> dict[str, Path]:
    metrics_dir = output_dir / "metrics"
    plots_dir = output_dir / "plots"
    videos_dir = output_dir / "videos"
    checkpoints_dir = output_dir / "checkpoints"

    metrics_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    if save_video:
        videos_dir.mkdir(parents=True, exist_ok=True)

    return {
        "output_dir": output_dir,
        "metrics_dir": metrics_dir,
        "plots_dir": plots_dir,
        "videos_dir": videos_dir,
        "checkpoints_dir": checkpoints_dir,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="", encoding="utf-8") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_series(
    x,
    y,
    title: str,
    xlabel: str,
    ylabel: str,
    path: Path,
    color: str = "tab:blue",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    plt.plot(x, y, color=color, linewidth=2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _to_float_array(values: list[Any]) -> np.ndarray:
    out: list[float] = []
    for v in values:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            out.append(np.nan)
    return np.asarray(out, dtype=np.float64)


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    window = max(1, int(window))
    out = np.full_like(values, np.nan, dtype=np.float64)
    for i in range(len(values)):
        start = max(0, i - window + 1)
        chunk = values[start:i + 1]
        finite = chunk[np.isfinite(chunk)]
        if finite.size:
            out[i] = float(np.mean(finite))
    return out


def _plot_training_dashboard(plots_dir: Path, train_metrics: list[dict]) -> None:
    if not train_metrics:
        return
    x = _to_float_array([m.get("iteration", i + 1) for i, m in enumerate(train_metrics)])
    reward = _to_float_array([m.get("episode_reward_mean") for m in train_metrics])
    p_loss = _to_float_array([m.get("policy_loss") for m in train_metrics])
    vf_loss = _to_float_array([m.get("vf_loss") for m in train_metrics])
    t_loss = _to_float_array([m.get("total_loss") for m in train_metrics])
    w = max(5, len(train_metrics) // 20)

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Training Dashboard", fontsize=14, fontweight="bold")

    axs[0, 0].plot(x, reward, color="tab:green", alpha=0.35, linewidth=1.5, label="Raw")
    axs[0, 0].plot(x, _moving_average(reward, w), color="tab:green", linewidth=2.2, label=f"Moving Avg ({w})")
    axs[0, 0].set_title("Reward Mean")
    axs[0, 0].set_xlabel("Iteration")
    axs[0, 0].set_ylabel("Reward")
    axs[0, 0].grid(True, alpha=0.3)
    axs[0, 0].legend()

    axs[0, 1].plot(x, p_loss, color="tab:blue", alpha=0.4, linewidth=1.5, label="Raw")
    axs[0, 1].plot(x, _moving_average(p_loss, w), color="tab:blue", linewidth=2.2, label=f"Moving Avg ({w})")
    axs[0, 1].set_title("Policy Loss")
    axs[0, 1].set_xlabel("Iteration")
    axs[0, 1].set_ylabel("Loss")
    axs[0, 1].grid(True, alpha=0.3)
    axs[0, 1].legend()

    axs[1, 0].plot(x, vf_loss, color="tab:orange", alpha=0.4, linewidth=1.5, label="Raw")
    axs[1, 0].plot(x, _moving_average(vf_loss, w), color="tab:orange", linewidth=2.2, label=f"Moving Avg ({w})")
    axs[1, 0].set_title("Value Loss")
    axs[1, 0].set_xlabel("Iteration")
    axs[1, 0].set_ylabel("Loss")
    axs[1, 0].grid(True, alpha=0.3)
    axs[1, 0].legend()

    axs[1, 1].plot(x, t_loss, color="tab:red", alpha=0.4, linewidth=1.5, label="Raw")
    axs[1, 1].plot(x, _moving_average(t_loss, w), color="tab:red", linewidth=2.2, label=f"Moving Avg ({w})")
    axs[1, 1].set_title("Total Loss")
    axs[1, 1].set_xlabel("Iteration")
    axs[1, 1].set_ylabel("Loss")
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].legend()

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(plots_dir / "train_dashboard.png", dpi=170)
    plt.close(fig)


def _plot_eval_dashboard(plots_dir: Path, eval_eps: list[dict]) -> None:
    if not eval_eps:
        return
    eps = _to_float_array([m.get("episode", i + 1) for i, m in enumerate(eval_eps)])
    survival = _to_float_array([m.get("escape_time_s", 0.0) for m in eval_eps])
    reward = _to_float_array([m.get("episode_reward", 0.0) for m in eval_eps])
    final_sep = _to_float_array([m.get("final_separation", 0.0) for m in eval_eps])
    escaped = _to_float_array([1.0 if bool(m.get("escaped", False)) else 0.0 for m in eval_eps])
    w = max(3, len(eval_eps) // 4)

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Evaluation Dashboard", fontsize=14, fontweight="bold")

    def add_eval_panel(ax, y: np.ndarray, title: str, ylabel: str, color: str) -> None:
        ax.plot(eps, y, marker="o", markersize=4, linewidth=1.6, color=color, alpha=0.5, label="Per Episode")
        ma = _moving_average(y, w)
        ax.plot(eps, ma, linewidth=2.4, color=color, label=f"Moving Avg ({w})")
        mean_val = float(np.nanmean(y)) if np.isfinite(y).any() else float("nan")
        if np.isfinite(mean_val):
            ax.axhline(mean_val, color="black", linestyle="--", linewidth=1.4, label=f"Overall Avg = {mean_val:.3f}")
        ax.set_title(title)
        ax.set_xlabel("Evaluation Episode")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()

    add_eval_panel(axs[0, 0], survival, "Survival Time", "Time [s]", "tab:purple")
    add_eval_panel(axs[0, 1], reward, "Episode Reward", "Reward", "tab:green")
    add_eval_panel(axs[1, 0], final_sep, "Final Separation", "Distance", "tab:blue")
    add_eval_panel(axs[1, 1], escaped, "Escape Outcome", "Escaped (1=yes, 0=no)", "tab:red")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(plots_dir / "eval_dashboard.png", dpi=170)
    plt.close(fig)


def save_metrics_and_plots(
    output_dir: Path,
    train_metrics: list[dict],
    eval_metrics: dict,
) -> None:
    metrics_dir = output_dir / "metrics"
    plots_dir = output_dir / "plots"

    write_json(metrics_dir / "train_metrics.json", train_metrics)
    write_csv(metrics_dir / "train_metrics.csv", train_metrics)
    write_json(metrics_dir / "eval_metrics.json", eval_metrics)
    write_csv(metrics_dir / "eval_episode_metrics.csv", eval_metrics.get("episodes", []))

    summary = {
        "escape_rate": eval_metrics.get("escape_rate"),
        "mean_escape_time_s": eval_metrics.get("mean_escape_time_s"),
        "mean_episode_reward": eval_metrics.get("mean_episode_reward"),
        "num_train_iterations": len(train_metrics),
        "num_eval_episodes": len(eval_metrics.get("episodes", [])),
    }
    write_json(metrics_dir / "summary.json", summary)

    _plot_training_dashboard(plots_dir=plots_dir, train_metrics=train_metrics)

    eval_eps = eval_metrics.get("episodes", [])
    _plot_eval_dashboard(plots_dir=plots_dir, eval_eps=eval_eps)
