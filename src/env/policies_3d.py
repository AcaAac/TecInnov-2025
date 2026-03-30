from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .config_loader_3d import Env3DConfig
from .kinematics_3d import normalize


def _boundary_avoidance_3d(pos: np.ndarray, config: Env3DConfig) -> np.ndarray:
    if not config.WALLS_MODE:
        return np.zeros(3, dtype=np.float64)

    arena_xy = float(config.ARENA_SIZE)
    arena_z = float(config.ARENA_HEIGHT)
    bounds = np.array([arena_xy, arena_xy, arena_z], dtype=np.float64)
    margin = 0.2 * np.array([arena_xy, arena_xy, arena_z], dtype=np.float64)
    gain = float(config.WALL_AVOIDANCE_GAIN)

    term = np.zeros(3, dtype=np.float64)
    for i in range(3):
        if pos[i] < margin[i]:
            term[i] += gain * (margin[i] - pos[i]) / max(margin[i], 1e-9)
        if pos[i] > bounds[i] - margin[i]:
            term[i] -= gain * (pos[i] - (bounds[i] - margin[i])) / max(margin[i], 1e-9)
    return term


def _random_perpendicular(vec: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    direction = normalize(vec)
    if np.linalg.norm(direction) < 1e-9:
        sample = rng.normal(size=3)
        return normalize(sample)

    sample = rng.normal(size=3)
    sample = sample - np.dot(sample, direction) * direction
    if np.linalg.norm(sample) < 1e-9:
        axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        if abs(direction[0]) > 0.9:
            axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        sample = np.cross(direction, axis)
    return normalize(sample)


class AgentPolicy:
    def get_action(self, obs: Any, agent_type: str):
        raise NotImplementedError


class RedPursuitPolicy3D(AgentPolicy):
    def __init__(self, config: Env3DConfig):
        self.config = config

    def get_action(self, obs, agent_type: str = "red"):
        my_pos = np.asarray(obs["red"][0:3], dtype=np.float64)
        target_pos = np.asarray(obs["blue"][0:3], dtype=np.float64)
        diff = target_pos - my_pos
        return self.config.V_RED_MAX * normalize(diff)


class BlueEvasivePolicy3D(AgentPolicy):
    def __init__(self, config: Env3DConfig, seed: Optional[int] = None):
        self.config = config
        self.rng = np.random.RandomState(config.SEED if seed is None else seed)

    def get_action(self, obs, agent_type: str = "blue"):
        my_pos = np.asarray(obs["blue"][0:3], dtype=np.float64)
        opp_pos = np.asarray(obs["red"][0:3], dtype=np.float64)

        rel = my_pos - opp_pos
        dist = np.linalg.norm(rel)
        escape_dir = normalize(rel)

        base_escape = escape_dir
        wall_term = _boundary_avoidance_3d(my_pos, self.config)

        juke_term = np.zeros(3, dtype=np.float64)
        if dist < self.config.JUKE_DISTANCE_THRESHOLD and np.linalg.norm(escape_dir) > 1e-9:
            perp = _random_perpendicular(escape_dir, self.rng)
            closeness = 1.0 - dist / max(self.config.JUKE_DISTANCE_THRESHOLD, 1e-9)
            juke_term = perp * max(0.0, closeness)

        vec = base_escape + wall_term + juke_term
        direction = normalize(vec)
        if np.linalg.norm(direction) < 1e-9:
            direction = escape_dir
        if np.linalg.norm(direction) < 1e-9:
            direction = normalize(self.rng.normal(size=3))
        return self.config.V_BLUE_MAX * direction
