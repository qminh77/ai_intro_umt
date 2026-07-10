"""Me cung va cac thuat toan tim duong can cho demo."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from heapq import heappop, heappush
from time import perf_counter
from typing import Callable, Iterable

import numpy as np


FREE = 0
WALL = 1
Position = tuple[int, int]
Heuristic = Callable[[Position, Position], float]

DIRECTIONS: tuple[Position, ...] = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
)


@dataclass(frozen=True)
class SearchResult:
    algorithm: str
    found: bool
    path: list[Position]
    path_length: int | None
    explored_nodes: int
    elapsed_ms: float
    explored_order: list[Position] = field(default_factory=list)


def generate_maze(
    height: int,
    width: int,
    wall_probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    return (rng.random((height, width)) < wall_probability).astype(np.int8)


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
        raise ValueError("Me cung khong co o trong.")
    row, col = cells[int(rng.integers(0, len(cells)))]
    return int(row), int(col)


def iter_neighbors(maze: np.ndarray, position: Position) -> Iterable[Position]:
    row, col = position
    for d_row, d_col in DIRECTIONS:
        candidate = (row + d_row, col + d_col)
        if is_free(maze, candidate):
            yield candidate


def manhattan(position: Position, goal: Position) -> float:
    return float(abs(position[0] - goal[0]) + abs(position[1] - goal[1]))


def reconstruct_path(
    came_from: dict[Position, Position],
    start: Position,
    goal: Position,
) -> list[Position]:
    if goal == start:
        return [start]
    if goal not in came_from:
        return []

    current = goal
    path = [current]
    while current != start:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def bfs_distances_to_goal(maze: np.ndarray, goal: Position) -> np.ndarray:
    """Chay BFS tu goal de tao nhan true_distance cho moi o co the di toi."""
    if not is_free(maze, goal):
        raise ValueError("Goal phai nam tren o trong.")

    distances = np.full(maze.shape, np.inf, dtype=float)
    distances[goal] = 0.0
    queue: deque[Position] = deque([goal])

    while queue:
        current = queue.popleft()
        for neighbor in iter_neighbors(maze, current):
            if np.isinf(distances[neighbor]):
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)

    return distances


def bfs_path(maze: np.ndarray, start: Position, goal: Position) -> SearchResult:
    started_at = perf_counter()
    if not is_free(maze, start) or not is_free(maze, goal):
        return SearchResult("BFS", False, [], None, 0, _elapsed_ms(started_at))

    queue: deque[Position] = deque([start])
    came_from: dict[Position, Position] = {}
    visited = {start}
    explored_order: list[Position] = []

    while queue:
        current = queue.popleft()
        explored_order.append(current)

        if current == goal:
            path = reconstruct_path(came_from, start, goal)
            return SearchResult(
                "BFS",
                True,
                path,
                len(path) - 1,
                len(explored_order),
                _elapsed_ms(started_at),
                explored_order,
            )

        for neighbor in iter_neighbors(maze, current):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)

    return SearchResult(
        "BFS", False, [], None, len(explored_order), _elapsed_ms(started_at), explored_order
    )


def astar(
    maze: np.ndarray,
    start: Position,
    goal: Position,
    heuristic: Heuristic,
    algorithm_name: str = "A*",
) -> SearchResult:
    started_at = perf_counter()
    if not is_free(maze, start) or not is_free(maze, goal):
        return SearchResult(algorithm_name, False, [], None, 0, _elapsed_ms(started_at))

    open_heap: list[tuple[float, int, Position]] = []
    heappush(open_heap, (float(heuristic(start, goal)), 0, start))
    came_from: dict[Position, Position] = {}
    g_score: dict[Position, float] = {start: 0.0}
    closed: set[Position] = set()
    explored_order: list[Position] = []
    counter = 0

    while open_heap:
        _, _, current = heappop(open_heap)
        if current in closed:
            continue

        closed.add(current)
        explored_order.append(current)

        if current == goal:
            path = reconstruct_path(came_from, start, goal)
            return SearchResult(
                algorithm_name,
                True,
                path,
                len(path) - 1,
                len(explored_order),
                _elapsed_ms(started_at),
                explored_order,
            )

        for neighbor in iter_neighbors(maze, current):
            tentative_g = g_score[current] + 1.0
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                counter += 1
                f_score = tentative_g + float(heuristic(neighbor, goal))
                heappush(open_heap, (f_score, counter, neighbor))

    return SearchResult(
        algorithm_name,
        False,
        [],
        None,
        len(explored_order),
        _elapsed_ms(started_at),
        explored_order,
    )


def _elapsed_ms(started_at: float) -> float:
    return (perf_counter() - started_at) * 1000
