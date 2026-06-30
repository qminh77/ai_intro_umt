from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from heapq import heappop, heappush
from time import perf_counter
from typing import Callable

import numpy as np

from .maze import Position, iter_neighbors, is_free


Heuristic = Callable[[Position, Position], float]


@dataclass(frozen=True)
class SearchResult:
    algorithm: str
    found: bool
    path: list[Position]
    path_length: int | None
    explored_nodes: int
    elapsed_ms: float


def manhattan(position: Position, goal: Position) -> float:
    return abs(position[0] - goal[0]) + abs(position[1] - goal[1])


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
    """Return shortest distance from every reachable cell to goal."""
    if not is_free(maze, goal):
        raise ValueError("Goal must be a free cell.")

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
    explored_nodes = 0

    while queue:
        current = queue.popleft()
        explored_nodes += 1
        if current == goal:
            path = reconstruct_path(came_from, start, goal)
            return SearchResult(
                "BFS",
                True,
                path,
                len(path) - 1,
                explored_nodes,
                _elapsed_ms(started_at),
            )

        for neighbor in iter_neighbors(maze, current):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)

    return SearchResult("BFS", False, [], None, explored_nodes, _elapsed_ms(started_at))


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
    heappush(open_heap, (heuristic(start, goal), 0, start))

    came_from: dict[Position, Position] = {}
    g_score: dict[Position, float] = {start: 0.0}
    closed: set[Position] = set()
    counter = 0
    explored_nodes = 0

    while open_heap:
        _, _, current = heappop(open_heap)
        if current in closed:
            continue

        closed.add(current)
        explored_nodes += 1

        if current == goal:
            path = reconstruct_path(came_from, start, goal)
            return SearchResult(
                algorithm_name,
                True,
                path,
                len(path) - 1,
                explored_nodes,
                _elapsed_ms(started_at),
            )

        for neighbor in iter_neighbors(maze, current):
            tentative_g = g_score[current] + 1.0
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + float(heuristic(neighbor, goal))
                counter += 1
                heappush(open_heap, (f_score, counter, neighbor))

    return SearchResult(algorithm_name, False, [], None, explored_nodes, _elapsed_ms(started_at))


def _elapsed_ms(started_at: float) -> float:
    return (perf_counter() - started_at) * 1000

