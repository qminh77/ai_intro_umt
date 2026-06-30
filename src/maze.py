from __future__ import annotations

from typing import Iterable

import numpy as np


FREE = 0
WALL = 1
Position = tuple[int, int]

DIRECTIONS: tuple[Position, ...] = (
    (-1, 0),  # up
    (1, 0),   # down
    (0, -1),  # left
    (0, 1),   # right
)


def generate_maze(
    height: int,
    width: int,
    wall_probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Create a random grid maze where 0 is path and 1 is wall."""
    maze = (rng.random((height, width)) < wall_probability).astype(np.int8)
    return maze


def in_bounds(maze: np.ndarray, position: Position) -> bool:
    row, col = position
    return 0 <= row < maze.shape[0] and 0 <= col < maze.shape[1]


def is_free(maze: np.ndarray, position: Position) -> bool:
    return in_bounds(maze, position) and maze[position] == FREE


def is_wall_or_outside(maze: np.ndarray, position: Position) -> bool:
    return not in_bounds(maze, position) or maze[position] == WALL


def free_cells(maze: np.ndarray) -> list[Position]:
    rows, cols = np.where(maze == FREE)
    return list(zip(rows.astype(int), cols.astype(int), strict=True))


def random_free_cell(maze: np.ndarray, rng: np.random.Generator) -> Position:
    cells = free_cells(maze)
    if not cells:
        raise ValueError("Maze has no free cells.")
    row, col = cells[int(rng.integers(0, len(cells)))]
    return int(row), int(col)


def iter_neighbors(maze: np.ndarray, position: Position) -> Iterable[Position]:
    row, col = position
    for d_row, d_col in DIRECTIONS:
        candidate = (row + d_row, col + d_col)
        if is_free(maze, candidate):
            yield candidate
