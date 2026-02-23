from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


@dataclass(frozen=True)
class EnvConfig:
    DT: float
    MAX_TIME: float
    ARENA_SIZE: float
    GRID_SIZE: int
    CAPTURE_RADIUS: float
    MIN_INIT_DIST: float
    WALLS_MODE: bool
    RED_MASS: float
    RED_MAX_ACCEL: float
    RED_MAX_SPEED: float
    RED_DRAG: float
    BLUE_MASS: float
    BLUE_MAX_ACCEL: float
    BLUE_MAX_SPEED: float
    BLUE_DRAG: float
    NUM_EPISODES: int
    OUTPUT_DIR: str
    SEED: int

    @property
    def MAX_STEPS(self) -> int:
        return int(self.MAX_TIME / self.DT)


_SCHEMA: Dict[str, type] = {
    "DT": float,
    "MAX_TIME": float,
    "ARENA_SIZE": float,
    "GRID_SIZE": int,
    "CAPTURE_RADIUS": float,
    "MIN_INIT_DIST": float,
    "WALLS_MODE": bool,
    "RED_MASS": float,
    "RED_MAX_ACCEL": float,
    "RED_MAX_SPEED": float,
    "RED_DRAG": float,
    "BLUE_MASS": float,
    "BLUE_MAX_ACCEL": float,
    "BLUE_MAX_SPEED": float,
    "BLUE_DRAG": float,
    "NUM_EPISODES": int,
    "OUTPUT_DIR": str,
    "SEED": int,
}


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def get_default_config_path(profile: str = "train") -> Path:
    profile_normalized = profile.lower().strip()
    if profile_normalized not in {"train", "simulate"}:
        raise ValueError(f"Unsupported profile '{profile}'. Expected 'train' or 'simulate'.")
    return _project_root() / "configs" / f"{profile_normalized}.yaml"


def _coerce_value(key: str, value: Any, expected_type: type) -> Any:
    if expected_type is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"Config key '{key}' must be a float.")
        return float(value)

    if expected_type is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"Config key '{key}' must be an int.")
        return int(value)

    if expected_type is bool:
        if not isinstance(value, bool):
            raise TypeError(f"Config key '{key}' must be a bool.")
        return value

    if expected_type is str:
        if not isinstance(value, str):
            raise TypeError(f"Config key '{key}' must be a string.")
        return value

    raise TypeError(f"Unsupported schema type for key '{key}'.")


def _validate_and_build(data: Dict[str, Any]) -> EnvConfig:
    missing = [key for key in _SCHEMA if key not in data]
    if missing:
        raise ValueError(f"Config missing required keys: {', '.join(missing)}")

    extra = [key for key in data if key not in _SCHEMA]
    if extra:
        raise ValueError(f"Config has unexpected keys: {', '.join(extra)}")

    normalized: Dict[str, Any] = {}
    for key, expected_type in _SCHEMA.items():
        normalized[key] = _coerce_value(key, data[key], expected_type)

    return EnvConfig(**normalized)


def load_env_config(
    profile: str = "train",
    config_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> EnvConfig:
    path = Path(config_path).expanduser().resolve() if config_path else get_default_config_path(profile)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise ValueError(f"Config file must contain a top-level mapping: {path}")

    merged: Dict[str, Any] = dict(raw)

    if overrides:
        unknown_override_keys = [key for key in overrides if key not in _SCHEMA]
        if unknown_override_keys:
            raise ValueError(
                f"Overrides contain unexpected keys: {', '.join(unknown_override_keys)}"
            )
        merged.update(overrides)

    return _validate_and_build(merged)
