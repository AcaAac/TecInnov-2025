from __future__ import annotations

from typing import Tuple

import numpy as np


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
    a = np.asarray(pos_a, dtype=np.float64).reshape(-1)
    b = np.asarray(pos_b, dtype=np.float64).reshape(-1)
    return float(np.linalg.norm(a - b))


def toroidal_displacement(
    target_pos: np.ndarray,
    my_pos: np.ndarray,
    arena_size: float,
    walls_mode: bool,
) -> np.ndarray:
    diff = np.asarray(target_pos, dtype=np.float64) - np.asarray(my_pos, dtype=np.float64)
    if walls_mode:
        return diff

    half = 0.5 * arena_size
    out = diff.copy()
    for i in range(2):
        if out[i] > half:
            out[i] -= arena_size
        elif out[i] < -half:
            out[i] += arena_size
    return out


def clamp_position_and_velocity(
    pos: np.ndarray,
    vel: np.ndarray,
    bounds: Tuple[float, ...],
    walls_mode: bool,
) -> tuple[np.ndarray, np.ndarray]:
    pos_out = np.asarray(pos, dtype=np.float64).reshape(-1).copy()
    vel_out = np.asarray(vel, dtype=np.float64).reshape(-1).copy()
    upper = np.asarray(bounds, dtype=np.float64).reshape(-1)

    dims = min(pos_out.shape[0], vel_out.shape[0], upper.shape[0])
    for i in range(dims):
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
