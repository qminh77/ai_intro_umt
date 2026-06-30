from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS, TARGET_COLUMN
from .maze import Position, is_wall_or_outside


def state_to_features(maze: np.ndarray, current: Position, goal: Position) -> np.ndarray:
    """Convert a maze state to the 10 ANN input features."""
    row, col = current
    goal_row, goal_col = goal
    height, width = maze.shape

    max_x = max(width - 1, 1)
    max_y = max(height - 1, 1)

    dx = abs(col - goal_col)
    dy = abs(row - goal_row)

    values = [
        col / max_x,
        row / max_y,
        goal_col / max_x,
        goal_row / max_y,
        dx / max_x,
        dy / max_y,
        float(is_wall_or_outside(maze, (row - 1, col))),
        float(is_wall_or_outside(maze, (row + 1, col))),
        float(is_wall_or_outside(maze, (row, col - 1))),
        float(is_wall_or_outside(maze, (row, col + 1))),
    ]
    return np.array(values, dtype=np.float32)


def build_dataset_rows(
    maze: np.ndarray,
    goal: Position,
    distances: np.ndarray,
    maze_id: int,
    max_samples: int | None,
    rng: np.random.Generator,
) -> list[dict[str, float]]:
    reachable_positions = [
        (int(row), int(col))
        for row, col in zip(*np.where(np.isfinite(distances)), strict=True)
    ]

    if max_samples is not None and len(reachable_positions) > max_samples:
        selected = rng.choice(len(reachable_positions), size=max_samples, replace=False)
        reachable_positions = [reachable_positions[int(index)] for index in selected]

    rows: list[dict[str, float]] = []
    for position in reachable_positions:
        features = state_to_features(maze, position, goal)
        row = {name: float(value) for name, value in zip(FEATURE_COLUMNS, features, strict=True)}
        row["maze_id"] = float(maze_id)
        row[TARGET_COLUMN] = float(distances[position])
        rows.append(row)

    return rows


def split_features_target(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = df[TARGET_COLUMN].to_numpy(dtype=np.float32)
    return x, y

