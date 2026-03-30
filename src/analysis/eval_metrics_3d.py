from __future__ import annotations

from typing import Callable, Dict, Iterable, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _safe_float(value) -> float:
    if value is None:
        return float("nan")
    return float(value)


def build_episode_summary_3d(
    step_df: pd.DataFrame,
    group_cols: Iterable[str] = ("policy_name", "episode_id"),
) -> pd.DataFrame:
    if step_df.empty:
        return pd.DataFrame()

    required_cols = {
        "time",
        "distance",
        "captured",
        "outcome",
        "init_blue_x",
        "init_blue_y",
        "init_blue_z",
        "init_red_x",
        "init_red_y",
        "init_red_z",
        "init_distance",
        "mode",
    }
    missing = required_cols - set(step_df.columns)
    if missing:
        raise ValueError(f"Missing required columns for 3D episode summary: {sorted(missing)}")

    summary_rows = []
    for _, group in step_df.groupby(list(group_cols), dropna=False, sort=False):
        group_sorted = group.sort_values("step")
        first = group_sorted.iloc[0]
        last = group_sorted.iloc[-1]

        captured = int(group_sorted["captured"].max())
        time_to_capture = float("nan")
        if captured:
            capture_rows = group_sorted[group_sorted["captured"] == 1]
            if not capture_rows.empty:
                time_to_capture = _safe_float(capture_rows.iloc[0]["time"])

        summary_rows.append(
            {
                "framework": first.get("framework", "unknown"),
                "policy_name": first.get("policy_name", "unknown"),
                "mode": first.get("mode", "unknown"),
                "episode_id": int(first.get("episode_id", 0)),
                "init_blue_x": _safe_float(first["init_blue_x"]),
                "init_blue_y": _safe_float(first["init_blue_y"]),
                "init_blue_z": _safe_float(first["init_blue_z"]),
                "init_red_x": _safe_float(first["init_red_x"]),
                "init_red_y": _safe_float(first["init_red_y"]),
                "init_red_z": _safe_float(first["init_red_z"]),
                "init_distance": _safe_float(first["init_distance"]),
                "outcome": str(last.get("outcome", "unknown")),
                "captured": captured,
                "evaded": 1 - captured,
                "time_to_capture": time_to_capture,
                "steps_to_terminal": int(last.get("step", len(group_sorted))),
                "terminal_time": _safe_float(last.get("time", float("nan"))),
                "min_distance": _safe_float(group_sorted["distance"].min()),
                "avg_distance": _safe_float(group_sorted["distance"].mean()),
                "terminal_distance": _safe_float(last["distance"]),
            }
        )

    return pd.DataFrame(summary_rows)


def add_initial_distance_bins(
    episode_df: pd.DataFrame,
    n_bins: int = 10,
    bin_col: str = "init_distance_bin",
) -> pd.DataFrame:
    if episode_df.empty:
        out = episode_df.copy()
        out[bin_col] = []
        return out

    out = episode_df.copy()
    distances = out["init_distance"].to_numpy(dtype=float)

    lo = np.nanmin(distances)
    hi = np.nanmax(distances)
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        out[bin_col] = "all"
        return out

    bins = np.linspace(lo, hi, n_bins + 1)
    out[bin_col] = pd.cut(out["init_distance"], bins=bins, include_lowest=True)
    return out


def compute_success_rates(episode_df: pd.DataFrame) -> pd.DataFrame:
    if episode_df.empty:
        return pd.DataFrame(columns=["policy_name", "capture_rate", "evade_rate", "n_episodes"])

    grouped = (
        episode_df.groupby("policy_name", as_index=False)
        .agg(capture_rate=("captured", "mean"), evade_rate=("evaded", "mean"), n_episodes=("episode_id", "count"))
        .sort_values("policy_name")
    )
    return grouped


def plot_success_vs_initial_distance(
    episode_df: pd.DataFrame,
    ax=None,
    n_bins: int = 10,
    title: str = "Success vs Initial Distance",
):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))

    if episode_df.empty:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return ax

    binned = add_initial_distance_bins(episode_df, n_bins=n_bins)

    for policy_name, group in binned.groupby("policy_name", sort=False):
        stats = (
            group.groupby("init_distance_bin", observed=False, as_index=False)
            .agg(capture_rate=("captured", "mean"), evade_rate=("evaded", "mean"), init_distance=("init_distance", "mean"))
            .sort_values("init_distance")
        )
        ax.plot(stats["init_distance"], stats["capture_rate"], marker="o", label=f"{policy_name} capture")
        ax.plot(stats["init_distance"], stats["evade_rate"], marker="x", linestyle="--", label=f"{policy_name} evade")

    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Initial Distance")
    ax.set_ylabel("Rate")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    return ax


def plot_time_to_capture_by_policy(
    episode_df: pd.DataFrame,
    ax=None,
    title: str = "Time to Capture Distribution",
):
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4.5))

    if episode_df.empty:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return ax

    data = []
    labels = []
    for policy_name, group in episode_df.groupby("policy_name", sort=False):
        vals = group.loc[group["captured"] == 1, "time_to_capture"].dropna().to_numpy()
        if vals.size:
            data.append(vals)
            labels.append(policy_name)

    if not data:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No capture events", ha="center", va="center", transform=ax.transAxes)
        return ax

    ax.boxplot(data, labels=labels, showfliers=False)
    ax.set_ylabel("Time to Capture (s)")
    ax.set_title(title)
    ax.grid(True, axis="y", alpha=0.3)
    return ax


def run_initial_condition_grid_3d(
    env,
    blue_policy,
    red_policy,
    grid_n: int = 9,
    fixed_red_pos: Optional[Tuple[float, float, float]] = None,
    fixed_blue_z: Optional[float] = None,
    max_steps: Optional[int] = None,
    include_trajectories: bool = True,
):
    if env.mode != "CONTINUOUS":
        raise ValueError("Initial-condition grid sweep is implemented for CONTINUOUS mode only.")

    cfg = env.config
    max_steps = cfg.MAX_STEPS if max_steps is None else int(max_steps)
    fixed_red = (
        np.array([cfg.ARENA_SIZE * 0.5, cfg.ARENA_SIZE * 0.5, cfg.ARENA_HEIGHT * 0.5], dtype=float)
        if fixed_red_pos is None
        else np.asarray(fixed_red_pos, dtype=float)
    )
    blue_z = cfg.ARENA_HEIGHT * 0.5 if fixed_blue_z is None else float(fixed_blue_z)

    xs = np.linspace(0.05 * cfg.ARENA_SIZE, 0.95 * cfg.ARENA_SIZE, grid_n)
    ys = np.linspace(0.05 * cfg.ARENA_SIZE, 0.95 * cfg.ARENA_SIZE, grid_n)

    regime_rows = []
    traj_map: Dict[Tuple[int, int], pd.DataFrame] = {}

    for ix, x in enumerate(xs):
        for iy, y in enumerate(ys):
            obs = env.reset(
                initial_blue_pos=np.array([x, y, blue_z], dtype=float),
                initial_red_pos=fixed_red,
                initial_blue_vel=np.zeros(3, dtype=float),
                initial_red_vel=np.zeros(3, dtype=float),
                skip_min_dist_check=True,
            )

            step_rows = []
            done = False
            while not done and env.step_count < max_steps:
                act_blue = blue_policy.get_action(obs, "blue")
                act_red = red_policy.get_action(obs, "red")
                obs, _, done, info = env.step(act_blue, act_red)

                step_rows.append(
                    {
                        "step": env.step_count,
                        "time": env.t,
                        "blue_x": float(obs["blue"][0]),
                        "blue_y": float(obs["blue"][1]),
                        "blue_z": float(obs["blue"][2]),
                        "red_x": float(obs["red"][0]),
                        "red_y": float(obs["red"][1]),
                        "red_z": float(obs["red"][2]),
                        "distance": float(info.get("distance", env.get_distance())),
                        "captured": int(info.get("caught", False)),
                        "outcome": info.get("outcome", "running"),
                    }
                )

            step_df = pd.DataFrame(step_rows)
            captured = int(step_df["captured"].max()) if not step_df.empty else 0
            outcome = "caught" if captured else "timeout"
            time_to_capture = float("nan")
            if captured and not step_df.empty:
                time_to_capture = float(step_df.loc[step_df["captured"] == 1, "time"].iloc[0])

            init_dist = float(np.linalg.norm(np.array([x, y, blue_z], dtype=float) - fixed_red))
            regime_rows.append(
                {
                    "grid_ix": ix,
                    "grid_iy": iy,
                    "init_blue_x": float(x),
                    "init_blue_y": float(y),
                    "init_blue_z": float(blue_z),
                    "init_red_x": float(fixed_red[0]),
                    "init_red_y": float(fixed_red[1]),
                    "init_red_z": float(fixed_red[2]),
                    "init_distance": init_dist,
                    "captured": captured,
                    "evaded": 1 - captured,
                    "outcome": outcome,
                    "time_to_capture": time_to_capture,
                    "steps_to_terminal": int(step_df["step"].max()) if not step_df.empty else 0,
                    "min_distance": float(step_df["distance"].min()) if not step_df.empty else init_dist,
                }
            )

            if include_trajectories and (ix, iy) in {
                (0, 0),
                (0, grid_n // 2),
                (0, grid_n - 1),
                (grid_n // 2, 0),
                (grid_n // 2, grid_n // 2),
                (grid_n // 2, grid_n - 1),
                (grid_n - 1, 0),
                (grid_n - 1, grid_n // 2),
                (grid_n - 1, grid_n - 1),
            }:
                traj_map[(ix, iy)] = step_df

    regime_df = pd.DataFrame(regime_rows)
    return regime_df, traj_map


def plot_regime_map_3d(
    regime_df: pd.DataFrame,
    arena_size: float,
    arena_height: float,
    ax=None,
    title: str = "Regime Map (fixed z)",
):
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 5.5))

    if regime_df.empty:
        ax.set_title(title)
        ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
        return ax

    pivot = regime_df.pivot(index="init_blue_y", columns="init_blue_x", values="captured")
    x = pivot.columns.to_numpy(dtype=float)
    y = pivot.index.to_numpy(dtype=float)
    z = pivot.to_numpy(dtype=float)

    im = ax.imshow(
        z,
        origin="lower",
        extent=[x.min(), x.max(), y.min(), y.max()],
        cmap="coolwarm",
        vmin=0,
        vmax=1,
        aspect="equal",
    )
    ax.scatter(
        regime_df["init_red_x"].iloc[0],
        regime_df["init_red_y"].iloc[0],
        c="black",
        s=40,
        marker="x",
        label="Fixed Red",
    )
    ax.set_xlim(0, arena_size)
    ax.set_ylim(0, arena_size)
    ax.set_xlabel("Blue init x")
    ax.set_ylabel("Blue init y")
    z_fixed = float(regime_df["init_blue_z"].iloc[0])
    ax.set_title(f"{title} (z={z_fixed:.2f}/{arena_height:.2f})")
    ax.legend(loc="upper right", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Capture (1=yes)")
    return ax


def plot_trajectory_projections(
    trajectory_map: Dict[Tuple[int, int], pd.DataFrame],
    arena_size: float,
    arena_height: float,
    fig=None,
    title: str = "Trajectory Projections",
):
    if fig is None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    else:
        axes = fig.subplots(1, 3)

    if not trajectory_map:
        for ax in np.ravel(axes):
            ax.set_title(title)
            ax.text(0.5, 0.5, "No trajectory samples", ha="center", va="center", transform=ax.transAxes)
        return fig, axes

    for traj in trajectory_map.values():
        if traj.empty:
            continue
        axes[0].plot(traj["blue_x"], traj["blue_y"], color="tab:blue", alpha=0.45, linewidth=1.0)
        axes[0].plot(traj["red_x"], traj["red_y"], color="tab:red", alpha=0.45, linewidth=1.0)
        axes[1].plot(traj["blue_x"], traj["blue_z"], color="tab:blue", alpha=0.45, linewidth=1.0)
        axes[1].plot(traj["red_x"], traj["red_z"], color="tab:red", alpha=0.45, linewidth=1.0)
        axes[2].plot(traj["blue_y"], traj["blue_z"], color="tab:blue", alpha=0.45, linewidth=1.0)
        axes[2].plot(traj["red_y"], traj["red_z"], color="tab:red", alpha=0.45, linewidth=1.0)

    axes[0].set_xlim(0, arena_size)
    axes[0].set_ylim(0, arena_size)
    axes[0].set_title("XY")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("y")

    axes[1].set_xlim(0, arena_size)
    axes[1].set_ylim(0, arena_height)
    axes[1].set_title("XZ")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("z")

    axes[2].set_xlim(0, arena_size)
    axes[2].set_ylim(0, arena_height)
    axes[2].set_title("YZ")
    axes[2].set_xlabel("y")
    axes[2].set_ylabel("z")

    for ax in axes:
        ax.grid(True, alpha=0.2)

    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return fig, axes


def plot_trajectory_scatter_3d(
    trajectory_map: Dict[Tuple[int, int], pd.DataFrame],
    arena_size: float,
    arena_height: float,
    ax=None,
    title: str = "Trajectory Samples (3D)",
):
    if ax is None:
        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(111, projection="3d")

    if not trajectory_map:
        ax.set_title(title)
        ax.text2D(0.5, 0.5, "No trajectory samples", transform=ax.transAxes)
        return ax

    for traj in trajectory_map.values():
        if traj.empty:
            continue
        ax.plot(traj["blue_x"], traj["blue_y"], traj["blue_z"], color="tab:blue", alpha=0.5, linewidth=1.0)
        ax.plot(traj["red_x"], traj["red_y"], traj["red_z"], color="tab:red", alpha=0.5, linewidth=1.0)

    ax.set_xlim(0, arena_size)
    ax.set_ylim(0, arena_size)
    ax.set_zlim(0, arena_height)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    return ax


def summarize_policy_eval_figure(
    episode_df: pd.DataFrame,
    output_path: Optional[str] = None,
    title_prefix: str = "Evaluation",
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    success = compute_success_rates(episode_df)
    if not success.empty:
        x = np.arange(len(success))
        w = 0.4
        axes[0, 0].bar(x - w / 2, success["capture_rate"], width=w, label="Capture rate", color="tab:red")
        axes[0, 0].bar(x + w / 2, success["evade_rate"], width=w, label="Evade rate", color="tab:blue")
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(success["policy_name"], rotation=20)
        axes[0, 0].set_ylim(0.0, 1.0)
        axes[0, 0].set_title("Outcome Rates")
        axes[0, 0].legend(fontsize=8)
        axes[0, 0].grid(True, axis="y", alpha=0.3)
    else:
        axes[0, 0].set_title("Outcome Rates")
        axes[0, 0].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[0, 0].transAxes)

    plot_time_to_capture_by_policy(episode_df, ax=axes[0, 1], title="Time-to-Capture")
    plot_success_vs_initial_distance(episode_df, ax=axes[1, 0], title="Success vs Init Distance")

    if not episode_df.empty:
        grouped = [g["min_distance"].dropna().to_numpy() for _, g in episode_df.groupby("policy_name", sort=False)]
        labels = [n for n, _ in episode_df.groupby("policy_name", sort=False)]
        if grouped:
            axes[1, 1].boxplot(grouped, labels=labels, showfliers=False)
            axes[1, 1].set_title("Min Distance by Policy")
            axes[1, 1].set_ylabel("Distance")
            axes[1, 1].grid(True, axis="y", alpha=0.3)
        else:
            axes[1, 1].set_title("Min Distance by Policy")
            axes[1, 1].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[1, 1].transAxes)
    else:
        axes[1, 1].set_title("Min Distance by Policy")
        axes[1, 1].text(0.5, 0.5, "No data", ha="center", va="center", transform=axes[1, 1].transAxes)

    fig.suptitle(f"{title_prefix} Metrics", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.97])

    if output_path:
        fig.savefig(output_path)
    return fig, axes
