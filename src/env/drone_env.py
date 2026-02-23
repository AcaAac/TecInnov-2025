from typing import Optional

import numpy as np

from .config_loader import EnvConfig, load_env_config
from .policies import AgentPolicy, RedPursuitPolicy


class DroneEnv:
    def __init__(
        self,
        mode: str = "CONTINUOUS",
        config: Optional[EnvConfig] = None,
        profile: str = "train",
        config_path: Optional[str] = None,
        config_overrides: Optional[dict] = None,
        red_policy: Optional[AgentPolicy] = None,
    ):
        self.mode = mode.upper()
        self.config = config or load_env_config(
            profile=profile,
            config_path=config_path,
            overrides=config_overrides,
        )

        self.t = 0.0
        self.step_count = 0
        self.blue_state = None
        self.red_state = None
        self.rng = np.random.RandomState(self.config.SEED)
        self.cell_size = self.config.ARENA_SIZE / self.config.GRID_SIZE
        self.red_policy = red_policy or RedPursuitPolicy(self.config)

    def seed(self, seed: int):
        self.rng = np.random.RandomState(seed)

    def reset(self):
        self.t = 0.0
        self.step_count = 0

        for _ in range(100):
            pos_blue = self.rng.rand(2) * self.config.ARENA_SIZE
            pos_red = self.rng.rand(2) * self.config.ARENA_SIZE

            if self.config.WALLS_MODE:
                dist = np.linalg.norm(pos_blue - pos_red)
            else:
                diff = pos_blue - pos_red
                for i in range(2):
                    if abs(diff[i]) > self.config.ARENA_SIZE / 2:
                        diff[i] = self.config.ARENA_SIZE - abs(diff[i])
                dist = np.linalg.norm(diff)

            if dist >= self.config.MIN_INIT_DIST:
                break

        if self.mode == "DISCRETE":
            self.blue_state = self._pos_to_idx(pos_blue)
            self.red_state = self._pos_to_idx(pos_red)
        else:
            self.blue_state = np.array([pos_blue[0], pos_blue[1], 0.0, 0.0], dtype=np.float64)
            self.red_state = np.array([pos_red[0], pos_red[1], 0.0, 0.0], dtype=np.float64)

        return self._get_obs()

    def _pos_to_idx(self, pos: np.ndarray) -> np.ndarray:
        idx = (pos / self.cell_size).astype(int)
        return np.clip(idx, 0, self.config.GRID_SIZE - 1)

    def _idx_to_pos(self, idx: np.ndarray) -> np.ndarray:
        return (idx + 0.5) * self.cell_size

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

        if self.mode == "DISCRETE":
            self.blue_state = self._step_discrete(self.blue_state, int(action_blue))
            self.red_state = self._step_discrete(self.red_state, int(action_red))
        else:
            self.blue_state = self._step_continuous(self.blue_state, action_blue, "blue")
            self.red_state = self._step_continuous(self.red_state, action_red, "red")

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

    def _step_continuous(self, state: np.ndarray, action, agent_type: str) -> np.ndarray:
        pos = state[0:2]
        vel = state[2:4]

        if agent_type == "red":
            max_acc = self.config.RED_MAX_ACCEL
            max_spd = self.config.RED_MAX_SPEED
            drag = self.config.RED_DRAG
            mass = self.config.RED_MASS
        else:
            max_acc = self.config.BLUE_MAX_ACCEL
            max_spd = self.config.BLUE_MAX_SPEED
            drag = self.config.BLUE_DRAG
            mass = self.config.BLUE_MASS

        action = np.asarray(action, dtype=np.float64)
        force = np.clip(action, -max_acc, max_acc)
        acc = force / mass

        vel += acc * self.config.DT
        vel *= 1.0 - drag * self.config.DT

        speed = np.linalg.norm(vel)
        if speed > max_spd:
            vel = vel / speed * max_spd

        pos += vel * self.config.DT

        if self.config.WALLS_MODE:
            for i in range(2):
                if pos[i] < 0:
                    pos[i] = 0
                    vel[i] = 0
                elif pos[i] > self.config.ARENA_SIZE:
                    pos[i] = self.config.ARENA_SIZE
                    vel[i] = 0
        else:
            for i in range(2):
                pos[i] = pos[i] % self.config.ARENA_SIZE

        return np.array([pos[0], pos[1], vel[0], vel[1]], dtype=np.float64)

    def get_distance(self) -> float:
        if self.mode == "DISCRETE":
            p1 = self._idx_to_pos(self.blue_state)
            p2 = self._idx_to_pos(self.red_state)
        else:
            p1 = self.blue_state[0:2]
            p2 = self.red_state[0:2]

        if self.config.WALLS_MODE:
            return float(np.linalg.norm(p1 - p2))

        diff = p1 - p2
        for i in range(2):
            if abs(diff[i]) > self.config.ARENA_SIZE / 2:
                diff[i] = self.config.ARENA_SIZE - abs(diff[i])
        return float(np.linalg.norm(diff))

    def _get_obs(self):
        return {
            "blue": self.blue_state.copy(),
            "red": self.red_state.copy(),
            "mode": self.mode,
        }

    def get_flat_state(self, obs=None) -> np.ndarray:
        if obs is None:
            obs = self._get_obs()

        if self.mode == "DISCRETE":
            b = 2.0 * (obs["blue"] / self.config.GRID_SIZE) - 1.0
            r = 2.0 * (obs["red"] / self.config.GRID_SIZE) - 1.0
            return np.concatenate([b, r], dtype=np.float32)

        b_pos = 2.0 * (obs["blue"][0:2] / self.config.ARENA_SIZE) - 1.0
        b_vel = obs["blue"][2:4] / self.config.BLUE_MAX_SPEED

        r_pos = 2.0 * (obs["red"][0:2] / self.config.ARENA_SIZE) - 1.0
        r_vel = obs["red"][2:4] / self.config.RED_MAX_SPEED

        return np.concatenate([b_pos, b_vel, r_pos, r_vel], dtype=np.float32)

    def get_state_dim(self) -> int:
        return 4 if self.mode == "DISCRETE" else 8

    def get_action_dim(self) -> int:
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
            p_blue = self._idx_to_pos(self.blue_state)
            p_red = self._idx_to_pos(self.red_state)
        else:
            p_blue = self.blue_state[0:2]
            p_red = self.red_state[0:2]

        blue_circle = patches.Circle(p_blue, radius=0.02, color="blue", label="Blue")
        red_circle = patches.Circle(p_red, radius=0.02, color="red", label="Red")
        capture_zone = patches.Circle(
            p_red,
            radius=self.config.CAPTURE_RADIUS,
            color="red",
            alpha=0.1,
        )

        ax.add_patch(blue_circle)
        ax.add_patch(red_circle)
        ax.add_patch(capture_zone)

        ax.set_title(f"Time: {self.t:.2f}s | Dist: {self.get_distance():.2f}")
        ax.legend(loc="upper right")
