from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..config_loader import EnvConfig


@dataclass
class PlanarVTOLDynamics:
    """Discrete-time planar linearized VTOL model around hover.

    State: x = [px, pz, vx, vz, theta, q]
    Input: u = [dT, tau]
    """

    m: float
    Iy: float
    g: float
    dt: float
    exact_discretization: bool = False

    def __post_init__(self):
        self.A = np.array(
            [
                [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, self.g, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        self.B = np.array(
            [
                [0.0, 0.0],
                [0.0, 0.0],
                [0.0, 0.0],
                [1.0 / self.m, 0.0],
                [0.0, 0.0],
                [0.0, 1.0 / self.Iy],
            ],
            dtype=np.float64,
        )

        self.Ad, self.Bd = self._discretize()

    def _discretize(self) -> tuple[np.ndarray, np.ndarray]:
        if self.exact_discretization:
            try:
                from scipy.linalg import expm

                n = self.A.shape[0]
                m = self.B.shape[1]
                aug = np.zeros((n + m, n + m), dtype=np.float64)
                aug[:n, :n] = self.A
                aug[:n, n:] = self.B
                exp_aug = expm(aug * self.dt)
                Ad = exp_aug[:n, :n]
                Bd = exp_aug[:n, n:]
                return Ad, Bd
            except Exception:
                pass

        Ad = np.eye(self.A.shape[0], dtype=np.float64) + self.A * self.dt
        Bd = self.B * self.dt
        return Ad, Bd

    def step(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        x = np.asarray(state, dtype=np.float64).reshape(6)
        u = np.asarray(control, dtype=np.float64).reshape(2)
        return self.Ad @ x + self.Bd @ u


@dataclass
class VelocityTrackingController:
    """Low-level controller mapping desired planar velocity to VTOL inputs."""

    m: float
    g: float
    theta_max: float
    dT_min: float
    dT_max: float
    tau_min: float
    tau_max: float
    Kv: float
    Ktheta: float
    Kq: float

    def compute_control(self, state: np.ndarray, v_des: np.ndarray) -> np.ndarray:
        x = np.asarray(state, dtype=np.float64).reshape(6)
        v_ref = np.asarray(v_des, dtype=np.float64).reshape(2)

        v_curr = x[2:4]
        theta = float(x[4])
        q = float(x[5])

        e_v = v_ref - v_curr
        a_des = self.Kv * e_v

        theta_des = np.clip(a_des[0] / self.g, -self.theta_max, self.theta_max)
        dT = np.clip(self.m * a_des[1], self.dT_min, self.dT_max)
        tau = self.Ktheta * (theta_des - theta) + self.Kq * (0.0 - q)
        tau = float(np.clip(tau, self.tau_min, self.tau_max))

        return np.array([dT, tau], dtype=np.float64)

    def clip_low_level(self, control: np.ndarray) -> np.ndarray:
        u = np.asarray(control, dtype=np.float64).reshape(2)
        u[0] = np.clip(u[0], self.dT_min, self.dT_max)
        u[1] = np.clip(u[1], self.tau_min, self.tau_max)
        return u


class VTOLDrone:
    """Per-drone abstraction for guidance/control/dynamics in continuous mode."""

    def __init__(
        self,
        name: str,
        config: EnvConfig,
        dynamics: PlanarVTOLDynamics,
        controller: VelocityTrackingController,
        v_max: float,
    ):
        self.name = name
        self.config = config
        self.dynamics = dynamics
        self.controller = controller
        self.v_max = float(v_max)
        self.state = np.zeros(6, dtype=np.float64)

    @property
    def pos(self) -> np.ndarray:
        return self.state[0:2]

    @property
    def vel(self) -> np.ndarray:
        return self.state[2:4]

    def set_state(self, state: np.ndarray):
        arr = np.asarray(state, dtype=np.float64).reshape(-1)
        if arr.shape[0] != 6:
            raise ValueError(f"{self.name} VTOL state must have 6 elements.")
        self.state = arr.copy()

    def control_step(self, desired_velocity: np.ndarray) -> np.ndarray:
        return self.controller.compute_control(self.state, desired_velocity)

    def dynamics_step(self, control: np.ndarray) -> np.ndarray:
        next_state = self.dynamics.step(self.state, control)

        if self.config.WALLS_MODE:
            for i in range(2):
                if next_state[i] < 0.0:
                    next_state[i] = 0.0
                    next_state[i + 2] = 0.0
                elif next_state[i] > self.config.ARENA_SIZE:
                    next_state[i] = self.config.ARENA_SIZE
                    next_state[i + 2] = 0.0
        else:
            for i in range(2):
                next_state[i] = next_state[i] % self.config.ARENA_SIZE

        self.state = next_state
        return self.state.copy()

    def clip_desired_velocity(self, desired_velocity: np.ndarray) -> np.ndarray:
        v = np.asarray(desired_velocity, dtype=np.float64).reshape(2)
        nrm = np.linalg.norm(v)
        if nrm <= self.v_max or nrm < 1e-9:
            return v
        return v * (self.v_max / nrm)

    def step(self, desired_velocity: Optional[np.ndarray] = None, low_level: Optional[np.ndarray] = None) -> np.ndarray:
        if (desired_velocity is None) == (low_level is None):
            raise ValueError("Provide exactly one of desired_velocity or low_level.")

        if desired_velocity is not None:
            v_des = self.clip_desired_velocity(desired_velocity)
            u = self.control_step(v_des)
        else:
            u = self.controller.clip_low_level(low_level)

        return self.dynamics_step(u)
