from typing import Any, Optional

import numpy as np

from .config_loader import EnvConfig
from .vtol import normalize, toroidal_displacement


def _get_toroidal_displacement(
    target_pos: np.ndarray,
    my_pos: np.ndarray,
    config: EnvConfig,
    mode: str = "CONTINUOUS",
) -> np.ndarray:
    if mode == "DISCRETE":
        diff = target_pos - my_pos
        if config.WALLS_MODE:
            return diff

        boundary = float(config.GRID_SIZE)
        half_boundary = boundary / 2.0
        for i in range(2):
            if diff[i] > half_boundary:
                diff[i] -= boundary
            elif diff[i] < -half_boundary:
                diff[i] += boundary
        return diff

    return toroidal_displacement(
        target_pos=target_pos,
        my_pos=my_pos,
        arena_size=config.ARENA_SIZE,
        walls_mode=config.WALLS_MODE,
    )


class AgentPolicy:
    def get_action(self, obs: Any, agent_type: str):
        raise NotImplementedError


class PursuerPolicy(AgentPolicy):
    def __init__(self, config: EnvConfig):
        self.config = config

    def get_action(self, obs, agent_type: str = "pursuer"):
        mode = obs["mode"]

        if mode == "DISCRETE":
            my_pos = obs["pursuer"]
            target_pos = obs["evader"]

            diff = _get_toroidal_displacement(target_pos, my_pos, self.config, mode="DISCRETE")
            dx = int(np.sign(diff[0]))
            dy = int(np.sign(diff[1]))

            mapping = {
                (0, 0): 0,
                (0, 1): 1,
                (0, -1): 2,
                (-1, 0): 3,
                (1, 0): 4,
                (-1, 1): 5,
                (1, 1): 6,
                (-1, -1): 7,
                (1, -1): 8,
            }
            return mapping.get((dx, dy), 0)

        # Option B guidance: output desired planar velocity toward the evader.
        my_state = obs["pursuer"]
        target_state = obs["evader"]
        pursuer_pos = np.asarray(my_state[0:2], dtype=np.float64)
        evader_pos = np.asarray(target_state[0:2], dtype=np.float64)

        to_evader = _get_toroidal_displacement(evader_pos, pursuer_pos, self.config, mode="CONTINUOUS")
        return self.config.V_PURSUER_MAX * normalize(to_evader)


class EvaderPolicy(AgentPolicy):
    def __init__(self, config: EnvConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.RandomState(config.SEED if seed is None else seed)

    def get_action(self, obs, agent_type: str = "evader"):
        mode = obs["mode"]

        if mode == "DISCRETE":
            my_pos = obs["evader"]
            opp_pos = obs["pursuer"]

            diff = _get_toroidal_displacement(my_pos, opp_pos, self.config, mode="DISCRETE")
            if abs(diff[0]) < 1 and abs(diff[1]) < 1:
                return self.rng.randint(1, 9)

            dx = int(np.sign(diff[0]))
            dy = int(np.sign(diff[1]))

            if self.config.WALLS_MODE:
                if my_pos[0] <= 1:
                    dx = 1
                if my_pos[0] >= self.config.GRID_SIZE - 2:
                    dx = -1
                if my_pos[1] <= 1:
                    dy = 1
                if my_pos[1] >= self.config.GRID_SIZE - 2:
                    dy = -1

            mapping = {
                (0, 0): 0,
                (0, 1): 1,
                (0, -1): 2,
                (-1, 0): 3,
                (1, 0): 4,
                (-1, 1): 5,
                (1, 1): 6,
                (-1, -1): 7,
                (1, -1): 8,
            }
            dx = np.clip(dx, -1, 1)
            dy = np.clip(dy, -1, 1)
            return mapping.get((dx, dy), 0)

        # Option B guidance: output desired planar velocity for evasion.
        my_state = obs["evader"]
        opp_state = obs["pursuer"]

        evader_pos = np.asarray(my_state[0:2], dtype=np.float64)
        pursuer_pos = np.asarray(opp_state[0:2], dtype=np.float64)

        rel = _get_toroidal_displacement(evader_pos, pursuer_pos, self.config, mode="CONTINUOUS")
        dist = np.linalg.norm(rel)
        escape_dir = normalize(rel)

        beta = 1.0
        base_escape = beta * escape_dir

        wall_term = np.zeros(2, dtype=np.float64)
        if self.config.WALLS_MODE:
            arena = float(self.config.ARENA_SIZE)
            margin = 0.2 * arena
            gain = float(self.config.WALL_AVOIDANCE_GAIN)
            if evader_pos[0] < margin:
                wall_term[0] += gain * (margin - evader_pos[0]) / max(margin, 1e-9)
            if evader_pos[0] > arena - margin:
                wall_term[0] -= gain * (evader_pos[0] - (arena - margin)) / max(margin, 1e-9)
            if evader_pos[1] < margin:
                wall_term[1] += gain * (margin - evader_pos[1]) / max(margin, 1e-9)
            if evader_pos[1] > arena - margin:
                wall_term[1] -= gain * (evader_pos[1] - (arena - margin)) / max(margin, 1e-9)

        juke_term = np.zeros(2, dtype=np.float64)
        if dist < self.config.JUKE_DISTANCE_THRESHOLD and np.linalg.norm(escape_dir) > 1e-9:
            perp = np.array([-escape_dir[1], escape_dir[0]], dtype=np.float64)
            if self.rng.rand() > 0.5:
                perp *= -1.0
            closeness = 1.0 - dist / max(self.config.JUKE_DISTANCE_THRESHOLD, 1e-9)
            juke_term = perp * max(0.0, closeness)

        vec = base_escape + wall_term + juke_term
        direction = normalize(vec)
        if np.linalg.norm(direction) < 1e-9:
            direction = escape_dir
        return self.config.V_EVADER_MAX * direction
