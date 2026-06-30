from __future__ import annotations

import numpy as np
import pandas as pd

from .features import build_dataset_rows
from .maze import Position, free_cells, generate_maze, random_free_cell
from .search import bfs_distances_to_goal


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
    all_rows: list[dict[str, float]] = []

    for maze_id in range(maze_count):
        maze, goal, distances = generate_training_maze(
            height=height,
            width=width,
            wall_probability=wall_probability,
            min_goal_distance=min_goal_distance,
            rng=rng,
        )
        all_rows.extend(
            build_dataset_rows(
                maze=maze,
                goal=goal,
                distances=distances,
                maze_id=maze_id,
                max_samples=max_samples_per_maze,
                rng=rng,
            )
        )

    df = pd.DataFrame(all_rows)
    return df.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def generate_training_maze(
    height: int,
    width: int,
    wall_probability: float,
    min_goal_distance: int,
    rng: np.random.Generator,
    max_attempts: int = 1000,
) -> tuple[np.ndarray, Position, np.ndarray]:
    """Generate a maze with a goal that has enough reachable training states."""
    for _ in range(max_attempts):
        maze = generate_maze(height, width, wall_probability, rng)
        cells = free_cells(maze)
        if len(cells) < min_goal_distance:
            continue

        goal = random_free_cell(maze, rng)
        distances = bfs_distances_to_goal(maze, goal)
        finite_distances = distances[np.isfinite(distances)]

        if len(finite_distances) < min_goal_distance:
            continue
        if finite_distances.max() < min_goal_distance:
            continue

        return maze, goal, distances

    raise RuntimeError(
        "Could not generate a suitable maze. Try lowering wall probability or min distance."
    )

