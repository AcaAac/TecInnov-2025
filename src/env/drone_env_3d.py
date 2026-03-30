from __future__ import annotations

from typing import Optional

import numpy as np

from .config_loader_3d import Env3DConfig, load_env_config_3d
from .kinematics_3d import KinematicDrone3D, euclidean_distance
from .policies_3d import AgentPolicy, BlueEvasivePolicy3D, RedPursuitPolicy3D


class DroneEnv3D:
    """Continuous 3D pursuit/evasion environment for drone-like kinematics."""

    def __init__(
        self,
        mode: str = "CONTINUOUS",
        config: Optional[Env3DConfig] = None,
        profile: str = "train",
        config_path: Optional[str] = None,
        config_overrides: Optional[dict] = None,
        red_policy: Optional[AgentPolicy] = None,
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
        self.blue_state = None
        self.red_state = None
        self.rng = np.random.RandomState(self.config.SEED)
        self.red_policy = red_policy or RedPursuitPolicy3D(self.config)

        self.blue_drone = KinematicDrone3D(
            name="blue",
            config=self.config,
            v_max=self.config.V_BLUE_MAX,
            accel_limit=self.config.ACCEL_BLUE_MAX,
        )
        self.red_drone = KinematicDrone3D(
            name="red",
            config=self.config,
            v_max=self.config.V_RED_MAX,
            accel_limit=self.config.ACCEL_RED_MAX,
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

    def _pair_distance(self, p_blue: np.ndarray, p_red: np.ndarray) -> float:
        return euclidean_distance(p_blue, p_red)

    def _sample_random_positions(self) -> tuple[np.ndarray, np.ndarray]:
        bounds = np.array([self.config.ARENA_SIZE, self.config.ARENA_SIZE, self.config.ARENA_HEIGHT], dtype=np.float64)
        for _ in range(200):
            pos_blue = self.rng.rand(3) * bounds
            pos_red = self.rng.rand(3) * bounds
            if self._pair_distance(pos_blue, pos_red) >= self.config.MIN_INIT_DIST:
                return pos_blue, pos_red
        return pos_blue, pos_red

    def reset(
        self,
        initial_blue_pos=None,
        initial_red_pos=None,
        initial_blue_vel=None,
        initial_red_vel=None,
        skip_min_dist_check: bool = False,
    ):
        self.t = 0.0
        self.step_count = 0

        sampled_blue, sampled_red = self._sample_random_positions()
        pos_blue = (
            self._continuous_pos(initial_blue_pos, "initial_blue_pos")
            if initial_blue_pos is not None
            else sampled_blue
        )
        pos_red = (
            self._continuous_pos(initial_red_pos, "initial_red_pos")
            if initial_red_pos is not None
            else sampled_red
        )
        vel_blue = (
            self._continuous_vel(initial_blue_vel, "initial_blue_vel")
            if initial_blue_vel is not None
            else np.zeros(3, dtype=np.float64)
        )
        vel_red = (
            self._continuous_vel(initial_red_vel, "initial_red_vel")
            if initial_red_vel is not None
            else np.zeros(3, dtype=np.float64)
        )

        if not skip_min_dist_check:
            sep = self._pair_distance(pos_blue, pos_red)
            if sep < self.config.MIN_INIT_DIST:
                raise ValueError("Provided 3D initial states violate MIN_INIT_DIST.")

        blue_init = np.array([pos_blue[0], pos_blue[1], pos_blue[2], vel_blue[0], vel_blue[1], vel_blue[2]], dtype=np.float64)
        red_init = np.array([pos_red[0], pos_red[1], pos_red[2], vel_red[0], vel_red[1], vel_red[2]], dtype=np.float64)

        self.blue_drone.set_state(blue_init)
        self.red_drone.set_state(red_init)
        self.blue_state = self.blue_drone.state.copy()
        self.red_state = self.red_drone.state.copy()
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

    def step(self, action_blue, action_red=None):
        obs_prev = self._get_obs()

        if action_red is None:
            action_red = self.red_policy.get_action(obs_prev, "red")
            red_action_source = "policy"
        else:
            red_action_source = "external"

        curr_dist = self.get_distance()
        caught = curr_dist <= self.config.CAPTURE_RADIUS
        if caught:
            return self._get_obs(), -10.0, True, {
                "outcome": "caught",
                "caught": True,
                "distance": curr_dist,
                "red_action_source": red_action_source,
            }

        if self.step_count >= self.config.MAX_STEPS:
            return self._get_obs(), 0.0, True, {
                "outcome": "timeout",
                "caught": False,
                "distance": curr_dist,
                "red_action_source": red_action_source,
            }

        act_blue = self._parse_continuous_action(action_blue)
        act_red = self._parse_continuous_action(action_red)

        self.blue_state = self.blue_drone.step(act_blue)
        self.red_state = self.red_drone.step(act_red)

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
            "red_action_source": red_action_source,
        }
        return self._get_obs(), reward, done, info

    def get_distance(self) -> float:
        return self._pair_distance(self.blue_state[0:3], self.red_state[0:3])

    def _get_obs(self):
        return {
            "blue": self.blue_state.copy(),
            "red": self.red_state.copy(),
            "mode": self.mode,
        }

    def get_flat_state(self, obs=None) -> np.ndarray:
        if obs is None:
            obs = self._get_obs()

        arena_xy = max(self.config.ARENA_SIZE, 1e-9)
        arena_z = max(self.config.ARENA_HEIGHT, 1e-9)

        b_pos = np.array(
            [
                2.0 * (obs["blue"][0] / arena_xy) - 1.0,
                2.0 * (obs["blue"][1] / arena_xy) - 1.0,
                2.0 * (obs["blue"][2] / arena_z) - 1.0,
            ],
            dtype=np.float32,
        )
        b_vel = np.asarray(obs["blue"][3:6], dtype=np.float32) / max(self.config.V_BLUE_MAX, 1e-9)

        r_pos = np.array(
            [
                2.0 * (obs["red"][0] / arena_xy) - 1.0,
                2.0 * (obs["red"][1] / arena_xy) - 1.0,
                2.0 * (obs["red"][2] / arena_z) - 1.0,
            ],
            dtype=np.float32,
        )
        r_vel = np.asarray(obs["red"][3:6], dtype=np.float32) / max(self.config.V_RED_MAX, 1e-9)

        return np.concatenate([b_pos, b_vel, r_pos, r_vel], dtype=np.float32)

    def get_state_dim(self) -> int:
        return 12

    def get_action_dim(self) -> int:
        return 3

    def render(self, ax=None):
        if ax is None:
            return

        blue_pos = self.blue_state[0:3]
        red_pos = self.red_state[0:3]

        ax.clear()
        ax.set_xlim(0, self.config.ARENA_SIZE)
        ax.set_ylim(0, self.config.ARENA_SIZE)
        if hasattr(ax, "set_zlim"):
            ax.set_zlim(0, self.config.ARENA_HEIGHT)
            ax.scatter([blue_pos[0]], [blue_pos[1]], [blue_pos[2]], c="blue", s=40, label="Blue")
            ax.scatter([red_pos[0]], [red_pos[1]], [red_pos[2]], c="red", s=40, label="Red")
            ax.plot([blue_pos[0], red_pos[0]], [blue_pos[1], red_pos[1]], [blue_pos[2], red_pos[2]], alpha=0.25, color="gray")
            ax.set_xlabel("x")
            ax.set_ylabel("y")
            ax.set_zlabel("z")
        else:
            ax.scatter([blue_pos[0]], [blue_pos[1]], c="blue", s=40, label="Blue")
            ax.scatter([red_pos[0]], [red_pos[1]], c="red", s=40, label="Red")
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        ax.set_title(f"Time: {self.t:.2f}s | Dist: {self.get_distance():.2f}")
        ax.legend(loc="upper right")
