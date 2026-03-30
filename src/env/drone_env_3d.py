from __future__ import annotations

from typing import Optional

import numpy as np

from .config_loader_3d import Env3DConfig, load_env_config_3d
from .kinematics_3d import KinematicDrone3D, euclidean_distance
from .policies_3d import AgentPolicy, EvaderPolicy3D, PursuerPolicy3D


def _draw_capture_wireframe(ax, center: np.ndarray, radius: float, color: str = "red") -> None:
    theta = np.linspace(0.0, 2.0 * np.pi, 16)
    phi = np.linspace(0.0, np.pi, 10)
    theta_grid, phi_grid = np.meshgrid(theta, phi)

    x = center[0] + radius * np.cos(theta_grid) * np.sin(phi_grid)
    y = center[1] + radius * np.sin(theta_grid) * np.sin(phi_grid)
    z = center[2] + radius * np.cos(phi_grid)

    ax.plot_wireframe(x, y, z, color=color, alpha=0.18, linewidth=0.6, rstride=1, cstride=1)


class DroneEnv3D:
    """Continuous 3D pursuit/evasion environment for drone-like kinematics."""

    def __init__(
        self,
        mode: str = "CONTINUOUS",
        config: Optional[Env3DConfig] = None,
        profile: str = "train",
        config_path: Optional[str] = None,
        config_overrides: Optional[dict] = None,
        pursuer_policy: Optional[AgentPolicy] = None,
    ):
        self.mode = mode.upper()
        if self.mode != "CONTINUOUS":
            raise ValueError("DroneEnv3D is continuous-only.")

        self.config = config or load_env_config_3d(
            profile=profile,
            config_path=config_path,
            overrides=config_overrides,
        )

        self.t = 0.0
        self.step_count = 0
        self.evader_state = None
        self.pursuer_state = None
        self.rng = np.random.RandomState(self.config.SEED)
        self.pursuer_policy = pursuer_policy or PursuerPolicy3D(self.config)

        self.evader_drone = KinematicDrone3D(
            name="evader",
            config=self.config,
            v_max=self.config.V_EVADER_MAX,
            accel_limit=self.config.ACCEL_EVADER_MAX,
        )
        self.pursuer_drone = KinematicDrone3D(
            name="pursuer",
            config=self.config,
            v_max=self.config.V_PURSUER_MAX,
            accel_limit=self.config.ACCEL_PURSUER_MAX,
        )

    def seed(self, seed: int):
        self.rng = np.random.RandomState(seed)

    def _continuous_pos(self, value, label: str) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.shape[0] != 3:
            raise ValueError(f"{label} must contain exactly three values.")
        if np.any(arr < 0.0) or np.any(arr > np.array([self.config.ARENA_SIZE, self.config.ARENA_SIZE, self.config.ARENA_HEIGHT])):
            raise ValueError(
                f"{label} must be inside the 3D arena bounds for all coordinates."
            )
        return arr

    def _continuous_vel(self, value, label: str) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.shape[0] != 3:
            raise ValueError(f"{label} must contain exactly three values.")
        return arr

    def _pair_distance(self, evader_pos: np.ndarray, pursuer_pos: np.ndarray) -> float:
        return euclidean_distance(evader_pos, pursuer_pos)

    def _sample_random_positions(self) -> tuple[np.ndarray, np.ndarray]:
        bounds = np.array([self.config.ARENA_SIZE, self.config.ARENA_SIZE, self.config.ARENA_HEIGHT], dtype=np.float64)
        for _ in range(200):
            pos_evader = self.rng.rand(3) * bounds
            pos_pursuer = self.rng.rand(3) * bounds
            if self._pair_distance(pos_evader, pos_pursuer) >= self.config.MIN_INIT_DIST:
                return pos_evader, pos_pursuer
        return pos_evader, pos_pursuer

    def reset(
        self,
        initial_evader_pos=None,
        initial_pursuer_pos=None,
        initial_evader_vel=None,
        initial_pursuer_vel=None,
        skip_min_dist_check: bool = False,
    ):
        self.t = 0.0
        self.step_count = 0

        sampled_evader, sampled_pursuer = self._sample_random_positions()
        pos_evader = (
            self._continuous_pos(initial_evader_pos, "initial_evader_pos")
            if initial_evader_pos is not None
            else sampled_evader
        )
        pos_pursuer = (
            self._continuous_pos(initial_pursuer_pos, "initial_pursuer_pos")
            if initial_pursuer_pos is not None
            else sampled_pursuer
        )
        vel_evader = (
            self._continuous_vel(initial_evader_vel, "initial_evader_vel")
            if initial_evader_vel is not None
            else np.zeros(3, dtype=np.float64)
        )
        vel_pursuer = (
            self._continuous_vel(initial_pursuer_vel, "initial_pursuer_vel")
            if initial_pursuer_vel is not None
            else np.zeros(3, dtype=np.float64)
        )

        if not skip_min_dist_check:
            sep = self._pair_distance(pos_evader, pos_pursuer)
            if sep < self.config.MIN_INIT_DIST:
                raise ValueError("Provided 3D initial states violate MIN_INIT_DIST.")

        evader_init = np.array([pos_evader[0], pos_evader[1], pos_evader[2], vel_evader[0], vel_evader[1], vel_evader[2]], dtype=np.float64)
        pursuer_init = np.array([pos_pursuer[0], pos_pursuer[1], pos_pursuer[2], vel_pursuer[0], vel_pursuer[1], vel_pursuer[2]], dtype=np.float64)

        self.evader_drone.set_state(evader_init)
        self.pursuer_drone.set_state(pursuer_init)
        self.evader_state = self.evader_drone.state.copy()
        self.pursuer_state = self.pursuer_drone.state.copy()
        return self._get_obs()

    def _parse_continuous_action(self, action) -> np.ndarray:
        if isinstance(action, dict):
            if "desired_velocity" in action:
                arr = np.asarray(action["desired_velocity"], dtype=np.float64).reshape(-1)
            elif "v_des" in action:
                arr = np.asarray(action["v_des"], dtype=np.float64).reshape(-1)
            else:
                raise ValueError("3D continuous action dict must include desired_velocity or v_des.")
        else:
            arr = np.asarray(action, dtype=np.float64).reshape(-1)

        if arr.shape[0] != 3:
            raise ValueError("3D continuous action must contain exactly three values.")
        if not np.all(np.isfinite(arr)):
            raise ValueError("3D continuous action must contain only finite values.")
        return arr

    def step(self, action_evader, action_pursuer=None):
        obs_prev = self._get_obs()

        if action_pursuer is None:
            action_pursuer = self.pursuer_policy.get_action(obs_prev, "pursuer")
            pursuer_action_source = "policy"
        else:
            pursuer_action_source = "external"

        curr_dist = self.get_distance()
        caught = curr_dist <= self.config.CAPTURE_RADIUS
        if caught:
            return self._get_obs(), -10.0, True, {
                "outcome": "caught",
                "caught": True,
                "distance": curr_dist,
                "pursuer_action_source": pursuer_action_source,
            }

        if self.step_count >= self.config.MAX_STEPS:
            return self._get_obs(), 0.0, True, {
                "outcome": "timeout",
                "caught": False,
                "distance": curr_dist,
                "pursuer_action_source": pursuer_action_source,
            }

        act_evader = self._parse_continuous_action(action_evader)
        act_pursuer = self._parse_continuous_action(action_pursuer)

        self.evader_state = self.evader_drone.step(act_evader)
        self.pursuer_state = self.pursuer_drone.step(act_pursuer)

        self.t += self.config.DT
        self.step_count += 1

        new_dist = self.get_distance()
        caught = new_dist <= self.config.CAPTURE_RADIUS
        done = caught or (self.step_count >= self.config.MAX_STEPS)

        reward = new_dist * 0.1
        if caught:
            reward -= 50.0
        else:
            reward += 0.05

        outcome = "caught" if caught else "timeout" if done else "running"
        info = {
            "outcome": outcome,
            "caught": caught,
            "distance": new_dist,
            "pursuer_action_source": pursuer_action_source,
        }
        return self._get_obs(), reward, done, info

    def get_distance(self) -> float:
        return self._pair_distance(self.evader_state[0:3], self.pursuer_state[0:3])

    def _get_obs(self):
        return {
            "evader": self.evader_state.copy(),
            "pursuer": self.pursuer_state.copy(),
            "mode": self.mode,
        }

    def get_flat_state(self, obs=None) -> np.ndarray:
        if obs is None:
            obs = self._get_obs()

        arena_xy = max(self.config.ARENA_SIZE, 1e-9)
        arena_z = max(self.config.ARENA_HEIGHT, 1e-9)

        evader_pos = np.array(
            [
                2.0 * (obs["evader"][0] / arena_xy) - 1.0,
                2.0 * (obs["evader"][1] / arena_xy) - 1.0,
                2.0 * (obs["evader"][2] / arena_z) - 1.0,
            ],
            dtype=np.float32,
        )
        evader_vel = np.asarray(obs["evader"][3:6], dtype=np.float32) / max(self.config.V_EVADER_MAX, 1e-9)

        pursuer_pos = np.array(
            [
                2.0 * (obs["pursuer"][0] / arena_xy) - 1.0,
                2.0 * (obs["pursuer"][1] / arena_xy) - 1.0,
                2.0 * (obs["pursuer"][2] / arena_z) - 1.0,
            ],
            dtype=np.float32,
        )
        pursuer_vel = np.asarray(obs["pursuer"][3:6], dtype=np.float32) / max(self.config.V_PURSUER_MAX, 1e-9)

        return np.concatenate([evader_pos, evader_vel, pursuer_pos, pursuer_vel], dtype=np.float32)

    def get_state_dim(self) -> int:
        return 12

    def get_action_dim(self) -> int:
        return 3

    def render(self, ax=None):
        if ax is None:
            return

        evader_pos = self.evader_state[0:3]
        pursuer_pos = self.pursuer_state[0:3]

        ax.clear()
        ax.set_xlim(0, self.config.ARENA_SIZE)
        ax.set_ylim(0, self.config.ARENA_SIZE)
        if hasattr(ax, "set_zlim"):
            ax.set_zlim(0, self.config.ARENA_HEIGHT)
            ax.scatter([evader_pos[0]], [evader_pos[1]], [evader_pos[2]], c="blue", s=40, label="Evader")
            ax.scatter([pursuer_pos[0]], [pursuer_pos[1]], [pursuer_pos[2]], c="red", s=40, label="Pursuer")
            ax.plot([evader_pos[0], pursuer_pos[0]], [evader_pos[1], pursuer_pos[1]], [evader_pos[2], pursuer_pos[2]], alpha=0.25, color="gray")
            _draw_capture_wireframe(ax, pursuer_pos, self.config.CAPTURE_RADIUS)
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
        else:
            ax.scatter([evader_pos[0]], [evader_pos[1]], c="blue", s=40, label="Evader")
            ax.scatter([pursuer_pos[0]], [pursuer_pos[1]], c="red", s=40, label="Pursuer")
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        ax.set_title(f"Time: {self.t:.2f}s | Dist: {self.get_distance():.2f}")
        ax.legend(loc="upper right")
