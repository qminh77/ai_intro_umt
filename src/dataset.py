"""Sinh dataset huan luyen ANN cho heuristic me cung."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR, FEATURE_COLUMNS, TARGET_COLUMN
from .maze import (
    Position,
    bfs_distances_to_goal,
    free_cells,
    generate_maze,
    is_wall_or_outside,
    random_free_cell,
)


def state_to_features(maze: np.ndarray, current: Position, goal: Position) -> np.ndarray:
    """Doi mot trang thai me cung thanh vector 10 dac trung cho ANN."""
    row, col = current
    goal_row, goal_col = goal
    height, width = maze.shape
    max_x = max(width - 1, 1)
    max_y = max(height - 1, 1)

    return np.array(
        [
            col / max_x,
            row / max_y,
            goal_col / max_x,
            goal_row / max_y,
            abs(col - goal_col) / max_x,
            abs(row - goal_row) / max_y,
            float(is_wall_or_outside(maze, (row - 1, col))),
            float(is_wall_or_outside(maze, (row + 1, col))),
            float(is_wall_or_outside(maze, (row, col - 1))),
            float(is_wall_or_outside(maze, (row, col + 1))),
        ],
        dtype=np.float32,
    )


def split_features_target(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    x = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = df[TARGET_COLUMN].to_numpy(dtype=np.float32)
    return x, y


def generate_labeled_dataset(
    maze_count: int,
    height: int,
    width: int,
    wall_probability: float,
    min_goal_distance: int,
    max_samples_per_maze: int | None,
    seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []

    for maze_id in range(maze_count):
        maze, goal, distances = generate_training_maze(
            height=height,
            width=width,
            wall_probability=wall_probability,
            min_goal_distance=min_goal_distance,
            rng=rng,
        )
        rows.extend(
            build_dataset_rows(
                maze=maze,
                goal=goal,
                distances=distances,
                maze_id=maze_id,
                max_samples=max_samples_per_maze,
                rng=rng,
            )
        )

    return pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def generate_training_maze(
    height: int,
    width: int,
    wall_probability: float,
    min_goal_distance: int,
    rng: np.random.Generator,
    max_attempts: int = 1000,
) -> tuple[np.ndarray, Position, np.ndarray]:
    for _ in range(max_attempts):
        maze = generate_maze(height, width, wall_probability, rng)
        if len(free_cells(maze)) < min_goal_distance:
            continue

        goal = random_free_cell(maze, rng)
        distances = bfs_distances_to_goal(maze, goal)
        finite_distances = distances[np.isfinite(distances)]

        if len(finite_distances) >= min_goal_distance and finite_distances.max() >= min_goal_distance:
            return maze, goal, distances

    raise RuntimeError(
        "Khong sinh duoc me cung hop le. Hay giam wall_probability hoac min_goal_distance."
    )


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
        values = state_to_features(maze, position, goal)
        row = {name: float(value) for name, value in zip(FEATURE_COLUMNS, values, strict=True)}
        row["maze_id"] = float(maze_id)
        row[TARGET_COLUMN] = float(distances[position])
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sinh dataset me cung duoc gan nhan bang BFS.")
    parser.add_argument("--mazes", type=int, default=500)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--wall-prob", type=float, default=0.25)
    parser.add_argument("--min-goal-distance", type=int, default=20)
    parser.add_argument("--max-samples-per-maze", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DATA_DIR / "maze_dataset.csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Sinh {args.mazes} me cung...")
    df = generate_labeled_dataset(
        maze_count=args.mazes,
        height=args.height,
        width=args.width,
        wall_probability=args.wall_prob,
        min_goal_distance=args.min_goal_distance,
        max_samples_per_maze=args.max_samples_per_maze,
        seed=args.seed,
    )
    df.to_csv(args.output, index=False)
    print(f"Da luu dataset: {args.output}")
    print(f"So dong: {len(df):,}")


if __name__ == "__main__":
    main()
