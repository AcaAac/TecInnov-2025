from __future__ import annotations

from pathlib import Path
from typing import Any

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from ray.rllib.algorithms.ppo import PPOConfig

from env import DoDFEnv


class PPOTrainer:
    def __init__(self, env_config: dict[str, Any] | None = None):
        self.env_config = env_config or {}
        self.dt = float(self.env_config.get("dt", 0.05))

        config = (
            PPOConfig()
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            .environment(env=DoDFEnv, env_config=self.env_config)
            .env_runners(num_env_runners=2)
            .training(
                lr=2e-4,
                train_batch_size_per_learner=2000,
                num_epochs=10,
            )
        )
        self.ppo = config.build_algo()

    def _inference_action(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        action = self.ppo.compute_single_action(obs, explore=not deterministic)
        if isinstance(action, tuple):
            action = action[0]
        return np.asarray(action, dtype=np.float32)

    @staticmethod
    def _set_equal_3d_axes(ax: Any, mins: np.ndarray, maxs: np.ndarray) -> float:
        """Use one world-unit scale on X, Y, and Z to prevent distortion."""
        center = 0.5 * (mins + maxs)
        side = float(max(np.max(maxs - mins), 1.0))
        half = 0.5 * side
        ax.set_xlim(center[0] - half, center[0] + half)
        ax.set_ylim(center[1] - half, center[1] + half)
        ax.set_zlim(center[2] - half, center[2] + half)
        ax.set_box_aspect((1.0, 1.0, 1.0))
        return side

    @staticmethod
    def _draw_winged_drone(
        ax: Any,
        position: np.ndarray,
        phi: float,
        theta: float,
        psi: float,
        color: str,
        size: float,
    ) -> None:
        """Draw an oriented aircraft glyph with visible main and tail wings."""
        cphi, sphi = np.cos(phi), np.sin(phi)
        cth, sth = np.cos(theta), np.sin(theta)
        cpsi, spsi = np.cos(psi), np.sin(psi)
        rotation = np.array(
            [
                [cth * cpsi, sphi * sth * cpsi - cphi * spsi, cphi * sth * cpsi + sphi * spsi],
                [cth * spsi, sphi * sth * spsi + cphi * cpsi, cphi * sth * spsi - sphi * cpsi],
                [-sth, sphi * cth, cphi * cth],
            ],
            dtype=np.float32,
        )

        # Body-frame line segments: fuselage, main wing, tail wing, vertical fin.
        segments = (
            ([-0.65, 0.0, 0.0], [0.85, 0.0, 0.0], 2.8),
            ([0.05, -0.90, 0.0], [0.05, 0.90, 0.0], 3.4),
            ([-0.48, -0.38, 0.0], [-0.48, 0.38, 0.0], 2.2),
            ([-0.48, 0.0, 0.0], [-0.66, 0.0, 0.38], 2.2),
        )
        for start, end, width in segments:
            points = np.vstack([start, end]) * size
            points = points @ rotation.T + position
            ax.plot(points[:, 0], points[:, 1], points[:, 2], color=color, linewidth=width)
        ax.scatter(*position, color=color, s=22)

    @staticmethod
    def _draw_quadcopter(
        ax: Any,
        position: np.ndarray,
        phi: float,
        theta: float,
        psi: float,
        color: str,
        size: float,
    ) -> None:
        """Draw the red rigid-body quadcopter with four visible rotor arms."""
        rotation = DoDFEnv._rotation_body_to_inertial(phi, theta, psi)
        arms = (
            ([-0.65, -0.65, 0.0], [0.65, 0.65, 0.0]),
            ([-0.65, 0.65, 0.0], [0.65, -0.65, 0.0]),
        )
        for start, end in arms:
            points = np.vstack([start, end]) * size
            points = points @ rotation.T + position
            ax.plot(points[:, 0], points[:, 1], points[:, 2], color=color, linewidth=3.0)
        rotor_points = np.array(
            [[-0.65, -0.65, 0.0], [-0.65, 0.65, 0.0],
             [0.65, -0.65, 0.0], [0.65, 0.65, 0.0]],
            dtype=np.float32,
        ) * size
        rotor_points = rotor_points @ rotation.T + position
        ax.scatter(rotor_points[:, 0], rotor_points[:, 1], rotor_points[:, 2],
                   facecolors="none", edgecolors=color, s=42, linewidths=2.0)
        # A short nose boom makes yaw orientation unambiguous.
        nose = np.array([[0.0, 0.0, 0.0], [0.9, 0.0, 0.0]], dtype=np.float32) * size
        nose = nose @ rotation.T + position
        ax.plot(nose[:, 0], nose[:, 1], nose[:, 2], color=color, linewidth=2.5)

    @staticmethod
    def _flatten(d: Any, prefix: str = "") -> dict[str, Any]:
        out: dict[str, Any] = {}
        if isinstance(d, dict):
            for k, v in d.items():
                key = f"{prefix}.{k}" if prefix else str(k)
                out.update(PPOTrainer._flatten(v, key))
        else:
            out[prefix] = d
        return out

    def _extract_losses(self, train_result: dict[str, Any]) -> dict[str, float | None]:
        flat = self._flatten(train_result)

        def pick(*names: str) -> float | None:
            for n in names:
                matches = [v for k, v in flat.items() if k.endswith(n)]
                if matches:
                    try:
                        return float(matches[0])
                    except (TypeError, ValueError):
                        continue
            return None

        return {
            "policy_loss": pick("policy_loss", "mean_policy_loss"),
            "vf_loss": pick("vf_loss", "value_loss", "mean_vf_loss"),
            "total_loss": pick("total_loss"),
            "entropy": pick("entropy", "mean_entropy"),
        }

    def _extract_episode_metrics(self, train_result: dict[str, Any]) -> tuple[float, float]:
        """Support both legacy and newer RLlib metric layouts."""
        flat = self._flatten(train_result)

        def pick_metric(*suffixes: str) -> float | None:
            for suffix in suffixes:
                for k, v in flat.items():
                    if k.endswith(suffix):
                        try:
                            return float(v)
                        except (TypeError, ValueError):
                            continue
            return None

        reward_mean = pick_metric(
            "episode_reward_mean",   # legacy
            "episode_return_mean",   # newer naming
        )
        len_mean = pick_metric("episode_len_mean")

        if reward_mean is None:
            reward_mean = float(np.nan)
        if len_mean is None:
            len_mean = float(np.nan)

        return reward_mean, len_mean

    def train(
        self,
        num_iterations: int = 10,
        episode_video_every: int | None = None,
        episode_video_dir: str | None = None,
        episode_video_fps: int = 12,
    ) -> list[dict[str, Any]]:
        history: list[dict[str, Any]] = []

        for i in range(1, num_iterations + 1):
            result = self.ppo.train()
            losses = self._extract_losses(result)
            episode_reward_mean, episode_len_mean = self._extract_episode_metrics(result)

            summary = {
                "iteration": i,
                "episode_reward_mean": episode_reward_mean,
                "episode_len_mean": episode_len_mean,
                **losses,
            }
            history.append(summary)
            print(
                f"[train {i:03d}] reward_mean={episode_reward_mean:.3f} "
                f"len_mean={episode_len_mean:.1f} "
                f"policy_loss={summary['policy_loss']} "
                f"vf_loss={summary['vf_loss']} "
                f"total_loss={summary['total_loss']} "
                f"entropy={summary['entropy']}"
            )
            if (
                episode_video_every
                and episode_video_every > 0
                and episode_video_dir
                and (i == 1 or i % episode_video_every == 0)
            ):
                video_path = self.save_single_episode_video(
                    output_path=str(Path(episode_video_dir) / f"episode_after_iter_{i:04d}.mp4"),
                    deterministic=True,
                    fps=episode_video_fps,
                )
                print(f"[train {i:03d}] saved episode video: {video_path}")

        return history

    @staticmethod
    def _save_episode_video(
        episode: int,
        learner_positions: list[np.ndarray],
        reference_positions: list[np.ndarray],
        output_dir: str,
        telemetry: list[dict[str, float]] | None = None,
        fps: int = 20,
        stride: int = 3,
    ) -> str:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"evaluation_episode_{episode:03d}.mp4"

        lp = np.asarray(learner_positions, dtype=np.float32)
        rp = np.asarray(reference_positions, dtype=np.float32)
        if len(lp) == 0 or len(rp) == 0:
            return ""

        # Fix one equal-scale camera for the complete episode. The padding is
        # based on the largest dimension so X, Y, and Z keep identical scales.
        all_pts = np.concatenate([lp, rp], axis=0)
        mins = all_pts.min(axis=0)
        maxs = all_pts.max(axis=0)
        fixed_side = max(float(np.max(maxs - mins)), 1.0)
        mins -= 0.08 * fixed_side
        maxs += 0.08 * fixed_side

        frames: list[np.ndarray] = []
        indices = list(range(1, len(lp), max(1, stride)))
        if indices[-1] != len(lp) - 1:
            indices.append(len(lp) - 1)

        for i in indices:
            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot(lp[: i + 1, 0], lp[: i + 1, 1], lp[: i + 1, 2], color="blue", linewidth=2, label="Evader")
            ax.plot(rp[: i + 1, 0], rp[: i + 1, 1], rp[: i + 1, 2], color="red", linewidth=2, label="Kamikaze")
            # No moving zoom: every frame uses the same full-episode bounds.
            side = PPOTrainer._set_equal_3d_axes(ax, mins, maxs)
            t = telemetry[i - 1] if telemetry and i - 1 < len(telemetry) else {}
            glyph_size = max(0.025 * side, 1.5)
            # Make blue easier to inspect and red visually smaller while keeping
            # both glyph sizes proportional to the fixed, equal-axis view.
            blue_glyph_size = 1.45 * glyph_size
            red_glyph_size = 0.60 * glyph_size
            # Winged glyphs expose heading, pitch, and bank relative to the scene.
            PPOTrainer._draw_winged_drone(
                ax, lp[i], t.get("blue_phi_rad", 0.0), t.get("blue_theta_rad", 0.0),
                t.get("blue_psi_rad", 0.0), "blue", blue_glyph_size,
            )
            PPOTrainer._draw_quadcopter(
                ax, rp[i], t.get("red_phi_rad", 0.0), t.get("red_theta_rad", 0.0),
                t.get("red_psi_rad", 0.0), "red", red_glyph_size,
            )
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title(f"Episode {episode:03d} - Evader vs Kamikaze")
            ax.legend(loc="upper right")
            ax.grid(True)
            if telemetry and i - 1 < len(telemetry):
                t = telemetry[i - 1]
                txt = (
                    f"LOS err: {t.get('los_err_deg', np.nan):6.2f} deg | "
                    f"yaw err: {t.get('yaw_err_deg', np.nan):6.2f} deg\n"
                    f"r_cmd: {t.get('r_cmd_deg_s', np.nan):6.2f} deg/s | "
                    f"r_real: {t.get('r_real_deg_s', np.nan):6.2f} deg/s\n"
                    f"v_cmd: {t.get('v_cmd_mps', np.nan):5.2f} m/s | "
                    f"v_real: {t.get('v_real_mps', np.nan):5.2f} m/s | "
                    f"opening: {t.get('opening_mps', np.nan):5.2f} m/s\n"
                    f"sat_cmd: {int(t.get('sat_cmd', 0.0))} | sat_real: {int(t.get('sat_real', 0.0))}"
                )
                ax.text2D(
                    0.02,
                    0.02,
                    txt,
                    transform=ax.transAxes,
                    fontsize=8.8,
                    bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "gray"},
                )

            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[1], fig.canvas.get_width_height()[0], 4)
            frames.append(frame[:, :, :3].copy())
            plt.close(fig)

        with imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=8) as writer:
            for f in frames:
                writer.append_data(f)
        return str(out_path)

    def evaluate(
        self,
        num_episodes: int = 10,
        deterministic: bool = True,
        save_video: bool = False,
        video_dir: str = "videos",
        video_fps: int = 12,
        video_episodes: int = 1,
    ) -> dict[str, Any]:
        episode_metrics: list[dict[str, Any]] = []
        env = DoDFEnv(config=self.env_config)

        for ep in range(1, num_episodes + 1):
            obs, _ = env.reset()
            done = False
            trunc = False
            steps = 0
            total_reward = 0.0
            learner_traj: list[np.ndarray] = [env.learner_pos.copy()]
            ref_traj: list[np.ndarray] = [env.ref_pos.copy()]
            episode_telemetry: list[dict[str, float]] = []

            while not (done or trunc):
                action = self._inference_action(obs=obs, deterministic=deterministic)
                obs, reward, done, trunc, info = env.step(action)
                total_reward += float(reward)
                steps += 1
                learner_traj.append(env.learner_pos.copy())
                ref_traj.append(env.ref_pos.copy())
                episode_telemetry.append(
                    {
                        "los_err_deg": float(np.rad2deg(info.get("telemetry_red_los_angle_error_rad", np.nan))),
                        "yaw_err_deg": float(np.rad2deg(info.get("telemetry_red_yaw_error_rad", np.nan))),
                        "r_cmd_deg_s": float(np.rad2deg(info.get("telemetry_red_r_cmd_rad_s", np.nan))),
                        "r_real_deg_s": float(np.rad2deg(info.get("telemetry_red_r_real_rad_s", np.nan))),
                        "v_cmd_mps": float(info.get("telemetry_red_speed_cmd_mps", np.nan)),
                        "v_real_mps": float(info.get("telemetry_red_speed_mps", np.nan)),
                        "opening_mps": float(info.get("telemetry_opening_speed_mps", np.nan)),
                        "sat_cmd": float(bool(info.get("telemetry_red_cmd_turn_saturated", False))),
                        "sat_real": float(bool(info.get("telemetry_red_real_turn_saturated", False))),
                        "blue_phi_rad": float(info.get("learner_phi_rad", 0.0)),
                        "blue_theta_rad": float(info.get("learner_theta_rad", 0.0)),
                        "blue_psi_rad": float(info.get("learner_psi_rad", 0.0)),
                        "red_phi_rad": float(info.get("reference_phi_rad", 0.0)),
                        "red_theta_rad": float(info.get("reference_theta_rad", 0.0)),
                        "red_psi_rad": float(info.get("reference_psi_rad", 0.0)),
                    }
                )

            escape_time = steps * self.dt
            escaped = bool(trunc and not done)
            caught = bool(info.get("caught", False))
            video_path = ""
            if save_video and ep <= max(0, int(video_episodes)):
                video_path = self._save_episode_video(
                    episode=ep,
                    learner_positions=learner_traj,
                    reference_positions=ref_traj,
                    output_dir=video_dir,
                    telemetry=episode_telemetry,
                    fps=video_fps,
                )

            metric = {
                "episode": ep,
                "escaped": escaped,
                "caught": caught,
                "escape_time_s": escape_time,
                "steps": steps,
                "episode_reward": total_reward,
                "final_separation": float(info.get("separation", np.nan)),
                "position_error_norm": float(info.get("position_error_norm", np.nan)),
                "velocity_error_norm": float(info.get("velocity_error_norm", np.nan)),
                "mean_red_los_angle_error_deg": float(np.mean([t["los_err_deg"] for t in episode_telemetry])) if episode_telemetry else float(np.nan),
                "mean_red_yaw_error_deg": float(np.mean([t["yaw_err_deg"] for t in episode_telemetry])) if episode_telemetry else float(np.nan),
                "mean_red_r_cmd_deg_s": float(np.mean([t["r_cmd_deg_s"] for t in episode_telemetry])) if episode_telemetry else float(np.nan),
                "mean_red_r_real_deg_s": float(np.mean([t["r_real_deg_s"] for t in episode_telemetry])) if episode_telemetry else float(np.nan),
                "mean_red_speed_cmd_mps": float(np.mean([t["v_cmd_mps"] for t in episode_telemetry])) if episode_telemetry else float(np.nan),
                "mean_red_speed_mps": float(np.mean([t["v_real_mps"] for t in episode_telemetry])) if episode_telemetry else float(np.nan),
                "mean_opening_speed_mps": float(np.mean([t["opening_mps"] for t in episode_telemetry])) if episode_telemetry else float(np.nan),
                "red_cmd_turn_saturation_ratio": float(np.mean([t["sat_cmd"] for t in episode_telemetry])) if episode_telemetry else float(np.nan),
                "red_real_turn_saturation_ratio": float(np.mean([t["sat_real"] for t in episode_telemetry])) if episode_telemetry else float(np.nan),
                "video_path": video_path,
            }
            episode_metrics.append(metric)
            print(
                f"[eval {ep:03d}] escaped={escaped} "
                f"caught={caught} "
                f"escape_time_s={escape_time:.2f} "
                f"reward={total_reward:.3f} "
                f"final_sep={metric['final_separation']:.2f} "
                f"video={video_path if video_path else 'none'}"
            )

        escape_rate = float(np.mean([m["escaped"] for m in episode_metrics])) if episode_metrics else 0.0
        mean_escape_time = float(np.mean([m["escape_time_s"] for m in episode_metrics])) if episode_metrics else 0.0
        mean_reward = float(np.mean([m["episode_reward"] for m in episode_metrics])) if episode_metrics else 0.0

        summary = {
            "escape_rate": escape_rate,
            "mean_escape_time_s": mean_escape_time,
            "mean_episode_reward": mean_reward,
            "episodes": episode_metrics,
        }
        print(
            f"[eval summary] escape_rate={escape_rate:.2%} "
            f"mean_escape_time_s={mean_escape_time:.2f} "
            f"mean_reward={mean_reward:.3f}"
        )
        return summary

    def save_single_episode_video(self, output_path: str, deterministic: bool = True, fps: int = 12) -> str:
        """Run one rollout and save a 3D trajectory video for quick qualitative inspection."""
        env = DoDFEnv(config=self.env_config)
        obs, _ = env.reset()
        done = False
        trunc = False
        learner_traj: list[np.ndarray] = [env.learner_pos.copy()]
        ref_traj: list[np.ndarray] = [env.ref_pos.copy()]
        episode_telemetry: list[dict[str, float]] = []

        while not (done or trunc):
            action = self._inference_action(obs=obs, deterministic=deterministic)
            obs, _, done, trunc, info = env.step(action)
            learner_traj.append(env.learner_pos.copy())
            ref_traj.append(env.ref_pos.copy())
            episode_telemetry.append(
                {
                    "los_err_deg": float(np.rad2deg(info.get("telemetry_red_los_angle_error_rad", np.nan))),
                    "yaw_err_deg": float(np.rad2deg(info.get("telemetry_red_yaw_error_rad", np.nan))),
                    "r_cmd_deg_s": float(np.rad2deg(info.get("telemetry_red_r_cmd_rad_s", np.nan))),
                    "r_real_deg_s": float(np.rad2deg(info.get("telemetry_red_r_real_rad_s", np.nan))),
                    "v_cmd_mps": float(info.get("telemetry_red_speed_cmd_mps", np.nan)),
                    "v_real_mps": float(info.get("telemetry_red_speed_mps", np.nan)),
                    "opening_mps": float(info.get("telemetry_opening_speed_mps", np.nan)),
                    "sat_cmd": float(bool(info.get("telemetry_red_cmd_turn_saturated", False))),
                    "sat_real": float(bool(info.get("telemetry_red_real_turn_saturated", False))),
                    "blue_phi_rad": float(info.get("learner_phi_rad", 0.0)),
                    "blue_theta_rad": float(info.get("learner_theta_rad", 0.0)),
                    "blue_psi_rad": float(info.get("learner_psi_rad", 0.0)),
                    "red_phi_rad": float(info.get("reference_phi_rad", 0.0)),
                    "red_theta_rad": float(info.get("reference_theta_rad", 0.0)),
                    "red_psi_rad": float(info.get("reference_psi_rad", 0.0)),
                }
            )

        lp = np.asarray(learner_traj, dtype=np.float32)
        rp = np.asarray(ref_traj, dtype=np.float32)
        # Compute fixed full-episode bounds once to avoid camera zoom or motion.
        all_pts = np.concatenate([lp, rp], axis=0)
        mins = all_pts.min(axis=0)
        maxs = all_pts.max(axis=0)
        fixed_side = max(float(np.max(maxs - mins)), 1.0)
        mins -= 0.08 * fixed_side
        maxs += 0.08 * fixed_side

        out_path = Path(output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frames: list[np.ndarray] = []
        indices = list(range(1, len(lp), 3))
        if not indices or indices[-1] != len(lp) - 1:
            indices.append(len(lp) - 1)

        for i in indices:
            fig = plt.figure(figsize=(7, 6))
            ax = fig.add_subplot(111, projection="3d")
            ax.plot(lp[: i + 1, 0], lp[: i + 1, 1], lp[: i + 1, 2], color="blue", linewidth=2, label="Evader")
            ax.plot(rp[: i + 1, 0], rp[: i + 1, 1], rp[: i + 1, 2], color="red", linewidth=2, label="Kamikaze")
            # Fixed camera with equal world-unit scaling on every axis.
            side = self._set_equal_3d_axes(ax, mins, maxs)
            t = episode_telemetry[i - 1] if i - 1 < len(episode_telemetry) else {}
            glyph_size = max(0.025 * side, 1.5)
            # Use distinct display scales without changing the physical models.
            blue_glyph_size = 1.45 * glyph_size
            red_glyph_size = 0.60 * glyph_size
            # Render both agents as oriented winged aircraft instead of points.
            self._draw_winged_drone(
                ax, lp[i], t.get("blue_phi_rad", 0.0), t.get("blue_theta_rad", 0.0),
                t.get("blue_psi_rad", 0.0), "blue", blue_glyph_size,
            )
            self._draw_quadcopter(
                ax, rp[i], t.get("red_phi_rad", 0.0), t.get("red_theta_rad", 0.0),
                t.get("red_psi_rad", 0.0), "red", red_glyph_size,
            )
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.set_title("Post-Training Episode - Evader vs Kamikaze")
            ax.legend(loc="upper right")
            ax.grid(True)
            if i - 1 < len(episode_telemetry):
                t = episode_telemetry[i - 1]
                txt = (
                    f"LOS err: {t.get('los_err_deg', np.nan):6.2f} deg | "
                    f"yaw err: {t.get('yaw_err_deg', np.nan):6.2f} deg\n"
                    f"r_cmd: {t.get('r_cmd_deg_s', np.nan):6.2f} deg/s | "
                    f"r_real: {t.get('r_real_deg_s', np.nan):6.2f} deg/s\n"
                    f"v_cmd: {t.get('v_cmd_mps', np.nan):5.2f} m/s | "
                    f"v_real: {t.get('v_real_mps', np.nan):5.2f} m/s | "
                    f"opening: {t.get('opening_mps', np.nan):5.2f} m/s\n"
                    f"sat_cmd: {int(t.get('sat_cmd', 0.0))} | sat_real: {int(t.get('sat_real', 0.0))}"
                )
                ax.text2D(
                    0.02,
                    0.02,
                    txt,
                    transform=ax.transAxes,
                    fontsize=8.8,
                    bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "gray"},
                )

            fig.canvas.draw()
            frame = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
            frame = frame.reshape(fig.canvas.get_width_height()[1], fig.canvas.get_width_height()[0], 4)
            frames.append(frame[:, :, :3].copy())
            plt.close(fig)

        with imageio.get_writer(str(out_path), fps=fps, codec="libx264", quality=8) as writer:
            for f in frames:
                writer.append_data(f)
        return str(out_path)

    def save(self, output_dir: str = "checkpoints") -> str:
        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        try:
            return self.ppo.save_to_path(out_dir.as_uri())
        except RuntimeError as e:
            # Old RLlib API stack does not support get_state() used by save_to_path().
            if "old API stack" in str(e):
                return str(self.ppo.save(str(out_dir)))
            raise
