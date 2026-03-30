from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .config_loader_3d import Env3DConfig


def normalize(vec: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    nrm = np.linalg.norm(arr)
    if nrm < eps:
        return np.zeros_like(arr, dtype=np.float64)
    return arr / nrm


def clip_norm(vec: np.ndarray, max_norm: float, eps: float = 1e-9) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64).reshape(-1)
    nrm = np.linalg.norm(arr)
    if nrm <= max_norm or nrm < eps:
        return arr
    return arr * (max_norm / nrm)


def euclidean_distance(pos_a: np.ndarray, pos_b: np.ndarray) -> float:
    a = np.asarray(pos_a, dtype=np.float64).reshape(3)
    b = np.asarray(pos_b, dtype=np.float64).reshape(3)
    return float(np.linalg.norm(a - b))


def clamp_position_and_velocity(
    pos: np.ndarray,
    vel: np.ndarray,
    bounds: Tuple[float, float, float],
    walls_mode: bool,
) -> tuple[np.ndarray, np.ndarray]:
    pos_out = np.asarray(pos, dtype=np.float64).reshape(3).copy()
    vel_out = np.asarray(vel, dtype=np.float64).reshape(3).copy()
    upper = np.asarray(bounds, dtype=np.float64).reshape(3)

    for i in range(3):
        if walls_mode:
            if pos_out[i] < 0.0:
                pos_out[i] = 0.0
                if vel_out[i] < 0.0:
                    vel_out[i] = 0.0
            elif pos_out[i] > upper[i]:
                pos_out[i] = upper[i]
                if vel_out[i] > 0.0:
                    vel_out[i] = 0.0
        else:
            pos_out[i] = float(np.clip(pos_out[i], 0.0, upper[i]))

    return pos_out, vel_out


@dataclass
class FirstOrder3DKinematics:
    dt: float
    tracking_gain: float
    accel_limit: float
    v_max: float
    bounds: Tuple[float, float, float]
    walls_mode: bool

    def step(self, state: np.ndarray, desired_velocity: np.ndarray) -> np.ndarray:
        x = np.asarray(state, dtype=np.float64).reshape(6)
        v_des = np.asarray(desired_velocity, dtype=np.float64).reshape(3)

        pos = x[0:3]
        vel = x[3:6]

        accel_cmd = self.tracking_gain * (v_des - vel)
        accel_cmd = clip_norm(accel_cmd, self.accel_limit)

        vel_next = vel + accel_cmd * self.dt
        vel_next = clip_norm(vel_next, self.v_max)
        pos_next = pos + vel_next * self.dt
        pos_next, vel_next = clamp_position_and_velocity(pos_next, vel_next, self.bounds, self.walls_mode)

        return np.concatenate([pos_next, vel_next], dtype=np.float64)


class KinematicDrone3D:
    def __init__(
        self,
        name: str,
        config: Env3DConfig,
        v_max: float,
        accel_limit: float,
    ):
        self.name = name
        self.config = config
        self.v_max = float(v_max)
        self.state = np.zeros(6, dtype=np.float64)
        self.dynamics = FirstOrder3DKinematics(
            dt=config.DT,
            tracking_gain=config.TRACKING_GAIN,
            accel_limit=float(accel_limit),
            v_max=self.v_max,
            bounds=(config.ARENA_SIZE, config.ARENA_SIZE, config.ARENA_HEIGHT),
            walls_mode=config.WALLS_MODE,
        )

    @property
    def pos(self) -> np.ndarray:
        return self.state[0:3]

    @property
    def vel(self) -> np.ndarray:
        return self.state[3:6]

    def set_state(self, state: np.ndarray):
        arr = np.asarray(state, dtype=np.float64).reshape(-1)
        if arr.shape[0] != 6:
            raise ValueError(f"{self.name} 3D state must have 6 elements.")
        self.state = arr.copy()

    def clip_desired_velocity(self, desired_velocity: np.ndarray) -> np.ndarray:
        return clip_norm(desired_velocity, self.v_max)

    def step(self, desired_velocity: Optional[np.ndarray] = None) -> np.ndarray:
        if desired_velocity is None:
            raise ValueError("Provide a desired_velocity command.")

        v_des = self.clip_desired_velocity(desired_velocity)
        self.state = self.dynamics.step(self.state, v_des)
        return self.state.copy()
