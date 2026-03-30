from __future__ import annotations

from typing import Optional

import numpy as np

from ..config_loader_3d import Env3DConfig
from .kinematics_3d import FirstOrder3DKinematics


class ThreeDimensionalVTOLDynamics:
    """VTOL-style 3D guidance backend built on velocity tracking.

    This keeps the 3D wrapper easy to swap while preserving a simple 6D
    position/velocity state for the learning pipeline.
    """

    def __init__(self, config: Env3DConfig, v_max: float, accel_limit: float):
        self.model = FirstOrder3DKinematics(
            dt=config.DT,
            tracking_gain=config.TRACKING_GAIN,
            accel_limit=float(accel_limit),
            v_max=float(v_max),
            bounds=(config.ARENA_SIZE, config.ARENA_SIZE, config.ARENA_HEIGHT),
            walls_mode=config.WALLS_MODE,
        )

    def step(self, state: np.ndarray, desired_velocity: np.ndarray) -> np.ndarray:
        return self.model.step(state, desired_velocity)


class VTOLDrone3D:
    """VTOL-flavored 3D drone wrapper.

    Swap this class in DroneEnv3D to change the backend used by the wrapper.
    """

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
        self.dynamics = ThreeDimensionalVTOLDynamics(
            config=config,
            v_max=self.v_max,
            accel_limit=accel_limit,
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
            raise ValueError(f"{self.name} VTOL3D state must have 6 elements.")
        self.state = arr.copy()

    def clip_desired_velocity(self, desired_velocity: np.ndarray) -> np.ndarray:
        v = np.asarray(desired_velocity, dtype=np.float64).reshape(3)
        nrm = np.linalg.norm(v)
        if nrm <= self.v_max or nrm < 1e-9:
            return v
        return v * (self.v_max / nrm)

    def step(self, desired_velocity: Optional[np.ndarray] = None) -> np.ndarray:
        if desired_velocity is None:
            raise ValueError("Provide a desired_velocity command.")

        v_des = self.clip_desired_velocity(desired_velocity)
        self.state = self.dynamics.step(self.state, v_des)
        return self.state.copy()
