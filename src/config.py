"""Cau hinh dung chung cho demo ANN + A* tren me cung."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


@dataclass(frozen=True)
class MazeConfig:
    width: int = 20
    height: int = 20
    wall_probability: float = 0.25
    min_goal_distance: int = 20
    seed: int = 42


FEATURE_COLUMNS = [
    "x",
    "y",
    "goal_x",
    "goal_y",
    "dx",
    "dy",
    "up_wall",
    "down_wall",
    "left_wall",
    "right_wall",
]
TARGET_COLUMN = "true_distance"
