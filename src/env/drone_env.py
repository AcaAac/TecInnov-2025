from typing import Optional

import numpy as np

from .config_loader import EnvConfig, load_env_config
from .policies import AgentPolicy, EvaderPolicy, PursuerPolicy
from .vtol import PlanarVTOLDynamics, VTOLDrone, VelocityTrackingController, toroidal_displacement


class DroneEnv:
    """Pursuit/evasion environment with DISCRETE and CONTINUOUS dynamics.

    DISCRETE mode keeps the grid-world transition model.
    CONTINUOUS mode uses a planar linearized VTOL backend per drone:
      1) high-level command (desired velocity or low-level VTOL input),
      2) low-level velocity-tracking control (if desired velocity provided),
      3) discrete linear VTOL propagation.
    """

    def __init__(
        self,
        mode: str = "CONTINUOUS",
        config: Optional[EnvConfig] = None,
        profile: str = "train",
        config_path: Optional[str] = None,
        config_overrides: Optional[dict] = None,
        pursuer_policy: Optional[AgentPolicy] = None,
    ):
        self.mode = mode.upper()
        self.config = config or load_env_config(
            profile=profile,
            config_path=config_path,
            overrides=config_overrides,
        )

        self.t = 0.0
        self.step_count = 0
        self.evader_state = None
        self.pursuer_state = None
        self.rng = np.random.RandomState(self.config.SEED)
        self.cell_size = self.config.ARENA_SIZE / self.config.GRID_SIZE
        self.pursuer_policy = pursuer_policy or PursuerPolicy(self.config)

        self._vtol_dynamics = None
        self._vtol_controller = None
        self.evader_drone: Optional[VTOLDrone] = None
        self.pursuer_drone: Optional[VTOLDrone] = None
        if self.mode == "CONTINUOUS":
            self._init_vtol_backend()

    def _init_vtol_backend(self):
        self._vtol_dynamics = PlanarVTOLDynamics(
            m=self.config.M,
            Iy=self.config.IY,
            g=self.config.G,
            dt=self.config.DT,
            exact_discretization=False,
        )
        self._vtol_controller = VelocityTrackingController(
            m=self.config.M,
            g=self.config.G,
            theta_max=self.config.THETA_MAX,
            dT_min=self.config.DT_MIN,
            dT_max=self.config.DT_MAX,
            tau_min=self.config.TAU_MIN,
            tau_max=self.config.TAU_MAX,
            Kv=self.config.KV,
            Ktheta=self.config.KTHETA,
            Kq=self.config.KQ,
        )
        self.evader_drone = VTOLDrone(
            name="evader",
            config=self.config,
            dynamics=self._vtol_dynamics,
            controller=self._vtol_controller,
            v_max=self.config.V_EVADER_MAX,
        )
        self.pursuer_drone = VTOLDrone(
            name="pursuer",
            config=self.config,
            dynamics=self._vtol_dynamics,
            controller=self._vtol_controller,
            v_max=self.config.V_PURSUER_MAX,
        )
    def seed(self, seed: int):
        self.rng = np.random.RandomState(seed)

    def _continuous_pos(self, value, label: str) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.shape[0] != 2:
            raise ValueError(f"{label} must contain exactly two values.")
        if np.any(arr < 0.0) or np.any(arr > self.config.ARENA_SIZE):
            raise ValueError(
                f"{label} must be inside [0, {self.config.ARENA_SIZE}] for both coordinates."
            )
        return arr

    def _continuous_vel(self, value, label: str) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.shape[0] != 2:
            raise ValueError(f"{label} must contain exactly two values.")
        return arr

    def _discrete_idx(self, value, label: str) -> np.ndarray:
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        if arr.shape[0] != 2:
            raise ValueError(f"{label} must contain exactly two values.")

        # If values look like continuous coordinates, map to grid indices.
        if np.any(np.abs(arr - np.round(arr)) > 1e-8):
            if np.any(arr < 0.0) or np.any(arr > self.config.ARENA_SIZE):
                raise ValueError(
                    f"{label} contains non-integer values out of arena bounds [0, {self.config.ARENA_SIZE}]."
                )
            return self._pos_to_idx(arr)

        idx = arr.astype(np.int64)
        if np.any(idx < 0) or np.any(idx >= self.config.GRID_SIZE):
            raise ValueError(
                f"{label} indices must be inside [0, {self.config.GRID_SIZE - 1}] for both coordinates."
            )
        return idx.astype(np.int64)

    def _pair_distance(self, evader_pos: np.ndarray, pursuer_pos: np.ndarray) -> float:
        if self.config.WALLS_MODE:
            return float(np.linalg.norm(evader_pos - pursuer_pos))

        diff = toroidal_displacement(
            target_pos=evader_pos,
            my_pos=pursuer_pos,
            arena_size=self.config.ARENA_SIZE,
            walls_mode=False,
        )
        return float(np.linalg.norm(diff))

    def _sample_random_positions(self) -> tuple:
        for _ in range(100):
            pos_evader = self.rng.rand(2) * self.config.ARENA_SIZE
            pos_pursuer = self.rng.rand(2) * self.config.ARENA_SIZE
            dist = self._pair_distance(pos_evader, pos_pursuer)
            if dist >= self.config.MIN_INIT_DIST:
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

        if self.mode == "DISCRETE":
            if initial_evader_vel is not None or initial_pursuer_vel is not None:
                raise ValueError("Velocities are not part of DISCRETE mode state.")

            sampled_evader, sampled_pursuer = self._sample_random_positions()
            evader_idx = (
                self._discrete_idx(initial_evader_pos, "initial_evader_pos")
                if initial_evader_pos is not None
                else self._pos_to_idx(sampled_evader)
            )
            pursuer_idx = (
                self._discrete_idx(initial_pursuer_pos, "initial_pursuer_pos")
                if initial_pursuer_pos is not None
                else self._pos_to_idx(sampled_pursuer)
            )

            if not skip_min_dist_check:
                evader_pos = self._idx_to_pos(evader_idx)
                pursuer_pos = self._idx_to_pos(pursuer_idx)
                sep = self._pair_distance(evader_pos, pursuer_pos)
                if sep < self.config.MIN_INIT_DIST:
                    raise ValueError("Provided discrete initial states violate MIN_INIT_DIST.")

            self.evader_state = evader_idx.astype(np.int64)
            self.pursuer_state = pursuer_idx.astype(np.int64)
        else:
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
                else np.zeros(2, dtype=np.float64)
            )
            vel_pursuer = (
                self._continuous_vel(initial_pursuer_vel, "initial_pursuer_vel")
                if initial_pursuer_vel is not None
                else np.zeros(2, dtype=np.float64)
            )

            if not skip_min_dist_check:
                sep = self._pair_distance(pos_evader, pos_pursuer)
                if sep < self.config.MIN_INIT_DIST:
                    raise ValueError("Provided continuous initial states violate MIN_INIT_DIST.")

            # VTOL state: [px, pz, vx, vz, theta, q]
            evader_init = np.array([pos_evader[0], pos_evader[1], vel_evader[0], vel_evader[1], 0.0, 0.0], dtype=np.float64)
            pursuer_init = np.array([pos_pursuer[0], pos_pursuer[1], vel_pursuer[0], vel_pursuer[1], 0.0, 0.0], dtype=np.float64)

            self.evader_drone.set_state(evader_init)
            self.pursuer_drone.set_state(pursuer_init)
            self.evader_state = self.evader_drone.state.copy()
            self.pursuer_state = self.pursuer_drone.state.copy()

        return self._get_obs()

    def _pos_to_idx(self, pos: np.ndarray) -> np.ndarray:
        idx = (pos / self.cell_size).astype(int)
        return np.clip(idx, 0, self.config.GRID_SIZE - 1)

    def _idx_to_pos(self, idx: np.ndarray) -> np.ndarray:
        return (idx + 0.5) * self.cell_size

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

        if self.mode == "DISCRETE":
            self.evader_state = self._step_discrete(self.evader_state, int(action_evader))
            self.pursuer_state = self._step_discrete(self.pursuer_state, int(action_pursuer))
        else:
            self.evader_state = self._step_continuous(self.evader_drone, action_evader)
            self.pursuer_state = self._step_continuous(self.pursuer_drone, action_pursuer)

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

    def _step_discrete(self, state: np.ndarray, action: int) -> np.ndarray:
        moves = [
            (0, 0),
            (0, 1),
            (0, -1),
            (-1, 0),
            (1, 0),
            (-1, 1),
            (1, 1),
            (-1, -1),
            (1, -1),
        ]
        dx, dy = moves[action]
        x, y = state
        if self.config.WALLS_MODE:
            nx, ny = np.clip([x + dx, y + dy], 0, self.config.GRID_SIZE - 1)
        else:
            nx = (x + dx) % self.config.GRID_SIZE
            ny = (y + dy) % self.config.GRID_SIZE
        return np.array([nx, ny])

    def _parse_continuous_action(self, action) -> tuple[str, np.ndarray]:
        """Parse continuous command as desired velocity or low-level VTOL input.

        Supported formats:
        - np.ndarray/list with shape (2,) -> desired planar velocity [vx_des, vz_des]
        - {"desired_velocity": [vx_des, vz_des]} or {"v_des": ...}
        - {"low_level": [dT, tau]} or {"u": ...}
        """
        if isinstance(action, dict):
            if "desired_velocity" in action:
                arr = np.asarray(action["desired_velocity"], dtype=np.float64).reshape(-1)
                kind = "desired_velocity"
            elif "v_des" in action:
                arr = np.asarray(action["v_des"], dtype=np.float64).reshape(-1)
                kind = "desired_velocity"
            elif "low_level" in action:
                arr = np.asarray(action["low_level"], dtype=np.float64).reshape(-1)
                kind = "low_level"
            elif "u" in action:
                arr = np.asarray(action["u"], dtype=np.float64).reshape(-1)
                kind = "low_level"
            else:
                raise ValueError("Continuous action dict must include desired_velocity/v_des or low_level/u.")
        else:
            arr = np.asarray(action, dtype=np.float64).reshape(-1)
            kind = "desired_velocity"

        if arr.shape[0] != 2:
            raise ValueError("Continuous action must contain exactly two values.")
        if not np.all(np.isfinite(arr)):
            raise ValueError("Continuous action must contain only finite values.")
        return kind, arr

    def _step_continuous(self, drone: VTOLDrone, action) -> np.ndarray:
        kind, cmd = self._parse_continuous_action(action)
        if kind == "desired_velocity":
            state = drone.step(desired_velocity=cmd)
        else:
            state = drone.step(low_level=cmd)
        return state

    def get_distance(self) -> float:
        if self.mode == "DISCRETE":
            evader_pos = self._idx_to_pos(self.evader_state)
            pursuer_pos = self._idx_to_pos(self.pursuer_state)
        else:
            evader_pos = self.evader_state[0:2]
            pursuer_pos = self.pursuer_state[0:2]

        return self._pair_distance(evader_pos, pursuer_pos)

    def _get_obs(self):
        return {
            "evader": self.evader_state.copy(),
            "pursuer": self.pursuer_state.copy(),
            "mode": self.mode,
        }

    def get_flat_state(self, obs=None) -> np.ndarray:
        if obs is None:
            obs = self._get_obs()

        evader_obs = obs["evader"]
        pursuer_obs = obs["pursuer"]

        if self.mode == "DISCRETE":
            b = 2.0 * (evader_obs / self.config.GRID_SIZE) - 1.0
            r = 2.0 * (pursuer_obs / self.config.GRID_SIZE) - 1.0
            return np.concatenate([b, r], dtype=np.float32)

        # Keep policy observation in position/velocity channels for compatibility.
        evader_pos = 2.0 * (evader_obs[0:2] / self.config.ARENA_SIZE) - 1.0
        evader_vel = evader_obs[2:4] / max(self.config.V_EVADER_MAX, 1e-9)

        pursuer_pos = 2.0 * (pursuer_obs[0:2] / self.config.ARENA_SIZE) - 1.0
        pursuer_vel = pursuer_obs[2:4] / max(self.config.V_PURSUER_MAX, 1e-9)

        return np.concatenate([evader_pos, evader_vel, pursuer_pos, pursuer_vel], dtype=np.float32)

    def get_state_dim(self) -> int:
        return 4 if self.mode == "DISCRETE" else 8

    def get_action_dim(self) -> int:
        # Continuous action = desired planar velocity by default.
        # Low-level VTOL command path has same dimensionality (2).
        return 9 if self.mode == "DISCRETE" else 2

    def render(self, ax=None):
        import matplotlib.patches as patches

        if ax is None:
            return

        ax.clear()
        ax.set_xlim(0, self.config.ARENA_SIZE)
        ax.set_ylim(0, self.config.ARENA_SIZE)
        ax.set_aspect("equal")

        if self.mode == "DISCRETE":
            evader_pos = self._idx_to_pos(self.evader_state)
            pursuer_pos = self._idx_to_pos(self.pursuer_state)
        else:
            evader_pos = self.evader_state[0:2]
            pursuer_pos = self.pursuer_state[0:2]

        evader_circle = patches.Circle(evader_pos, radius=0.02, color="blue", label="Evader")
        pursuer_circle = patches.Circle(pursuer_pos, radius=0.02, color="red", label="Pursuer")
        capture_zone = patches.Circle(
            pursuer_pos,
            radius=self.config.CAPTURE_RADIUS,
            color="red",
            alpha=0.1,
        )

        ax.add_patch(evader_circle)
        ax.add_patch(pursuer_circle)
        ax.add_patch(capture_zone)

        ax.set_title(f"Time: {self.t:.2f}s | Dist: {self.get_distance():.2f}")
        ax.legend(loc="upper right")
