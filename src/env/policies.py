from typing import Any, Optional

import numpy as np

from .config_loader import EnvConfig


def _get_toroidal_displacement(
    target_pos: np.ndarray,
    my_pos: np.ndarray,
    config: EnvConfig,
    mode: str = "CONTINUOUS",
) -> np.ndarray:
    diff = target_pos - my_pos

    if config.WALLS_MODE:
        return diff

    boundary = config.GRID_SIZE if mode == "DISCRETE" else config.ARENA_SIZE
    half_boundary = boundary / 2.0

    for i in range(2):
        if diff[i] > half_boundary:
            diff[i] -= boundary
        elif diff[i] < -half_boundary:
            diff[i] += boundary

    return diff


class AgentPolicy:
    def get_action(self, obs: Any, agent_type: str):
        raise NotImplementedError


class RedPursuitPolicy(AgentPolicy):
    def __init__(self, config: EnvConfig):
        self.config = config

    def get_action(self, obs, agent_type: str = "red"):
        mode = obs["mode"]

        if mode == "DISCRETE":
            my_pos = obs["red"]
            target_pos = obs["blue"]

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

        my_state = obs["red"]
        target_state = obs["blue"]

        p_red = my_state[0:2]
        v_red = my_state[2:4]
        p_blue = target_state[0:2]
        v_blue = target_state[2:4]

        error_pos = _get_toroidal_displacement(p_blue, p_red, self.config, mode="CONTINUOUS")
        error_vel = v_blue - v_red
        kp = 7.0
        kd = 1.0
        return kp * error_pos + kd * error_vel


class BlueEvasivePolicy(AgentPolicy):
    def __init__(self, config: EnvConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.RandomState(config.SEED if seed is None else seed)

    def get_action(self, obs, agent_type: str = "blue"):
        mode = obs["mode"]

        if mode == "DISCRETE":
            my_pos = obs["blue"]
            opp_pos = obs["red"]

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

        my_state = obs["blue"]
        opp_state = obs["red"]

        p_blue = my_state[0:2]
        p_red = opp_state[0:2]

        diff = _get_toroidal_displacement(p_blue, p_red, self.config, mode="CONTINUOUS")
        dist = np.linalg.norm(diff) + 1e-6
        repulse_dir = diff / dist
        force_red = repulse_dir * (0.5 / dist)

        force_wall = np.zeros(2)
        if self.config.WALLS_MODE:
            margin = 0.2
            if p_blue[0] < margin:
                force_wall[0] += 1.0 / (p_blue[0] + 0.1)
            if p_blue[0] > 1.0 - margin:
                force_wall[0] -= 1.0 / (1.0 - p_blue[0] + 0.1)
            if p_blue[1] < margin:
                force_wall[1] += 1.0 / (p_blue[1] + 0.1)
            if p_blue[1] > 1.0 - margin:
                force_wall[1] -= 1.0 / (1.0 - p_blue[1] + 0.1)

        juke_force = np.zeros(2)
        if dist < 0.25:
            perp = np.array([-repulse_dir[1], repulse_dir[0]])
            if self.rng.rand() > 0.5:
                perp *= -1
            juke_force = perp * 2.5

        return force_red + force_wall * 0.1 + juke_force
