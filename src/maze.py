"""Mê cung và các thuật toán tìm đường cần cho demo.

File này gom phần AI truyền thống: sinh mê cung, BFS, A* và heuristic
Manhattan. ANN chưa nằm ở đây; ANN chỉ được truyền vào A* dưới dạng một
hàm heuristic ở `demo.py`.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from heapq import heappop, heappush
from time import perf_counter
from typing import Callable, Iterable

import numpy as np


# Quy ước biểu diễn mê cung: 0 là ô trống, 1 là tường.
FREE = 0
WALL = 1

# Position luôn có dạng (row, col), tức là (dòng, cột).
Position = tuple[int, int]

# Heuristic là một hàm nhận vị trí hiện tại và goal, trả về chi phí ước lượng.
Heuristic = Callable[[Position, Position], float]

DIRECTIONS: tuple[Position, ...] = (
    (-1, 0),  # Lên.
    (1, 0),   # Xuống.
    (0, -1),  # Trái.
    (0, 1),   # Phải.
)


@dataclass(frozen=True)
class SearchResult:
    """Kết quả chuẩn hóa để BFS, A* Manhattan và A* ANN dễ so sánh."""

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
    # Mỗi ô được random độc lập. Nếu số random < wall_probability thì ô đó là tường.
    return (rng.random((height, width)) < wall_probability).astype(np.int8)


def in_bounds(maze: np.ndarray, position: Position) -> bool:
    # Kiểm tra vị trí có nằm trong biên ma trận hay không.
    row, col = position
    return 0 <= row < maze.shape[0] and 0 <= col < maze.shape[1]


def is_free(maze: np.ndarray, position: Position) -> bool:
    # Một ô đi được khi vừa nằm trong biên vừa không phải tường.
    return in_bounds(maze, position) and maze[position] == FREE


def is_wall_or_outside(maze: np.ndarray, position: Position) -> bool:
    # Dùng cho feature cảm biến tường xung quanh của ANN.
    return not in_bounds(maze, position) or maze[position] == WALL


def free_cells(maze: np.ndarray) -> list[Position]:
    # Lấy toàn bộ ô trống để chọn ngẫu nhiên start/goal hoặc kiểm tra mê cung hợp lệ.
    rows, cols = np.where(maze == FREE)
    return list(zip(rows.astype(int), cols.astype(int), strict=True))


def random_free_cell(maze: np.ndarray, rng: np.random.Generator) -> Position:
    # Chọn một ô trống bất kỳ làm goal hoặc start.
    cells = free_cells(maze)
    if not cells:
        raise ValueError("Mê cung không có ô trống.")
    row, col = cells[int(rng.integers(0, len(cells)))]
    return int(row), int(col)


def iter_neighbors(maze: np.ndarray, position: Position) -> Iterable[Position]:
    # Sinh ra các ô kề hợp lệ theo 4 hướng. Thuật toán BFS/A* đều dùng hàm này.
    row, col = position
    for d_row, d_col in DIRECTIONS:
        candidate = (row + d_row, col + d_col)
        if is_free(maze, candidate):
            yield candidate


def manhattan(position: Position, goal: Position) -> float:
    # Heuristic thủ công truyền thống: bỏ qua tường, chỉ tính lệch dòng + lệch cột.
    return float(abs(position[0] - goal[0]) + abs(position[1] - goal[1]))


def reconstruct_path(
    came_from: dict[Position, Position],
    start: Position,
    goal: Position,
) -> list[Position]:
    # came_from lưu "node này đi từ node nào". Ta lần ngược từ goal về start.
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
    """Chạy BFS từ goal để tạo nhãn `true_distance` cho mọi ô đi tới được."""

    if not is_free(maze, goal):
        raise ValueError("Goal phải nằm trên ô trống.")

    # Ban đầu mọi ô chưa biết khoảng cách nên đặt là vô cực.
    distances = np.full(maze.shape, np.inf, dtype=float)
    distances[goal] = 0.0
    queue: deque[Position] = deque([goal])

    while queue:
        current = queue.popleft()
        for neighbor in iter_neighbors(maze, current):
            # BFS đảm bảo lần đầu chạm tới một ô là khoảng cách ngắn nhất.
            if np.isinf(distances[neighbor]):
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)

    return distances


def bfs_path(maze: np.ndarray, start: Position, goal: Position) -> SearchResult:
    # BFS là tìm kiếm mù: không dùng heuristic, duyệt theo từng lớp khoảng cách.
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
            # Khi gặp goal, dựng lại đường đi bằng bảng came_from.
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
    # A* chọn node có f(n) = g(n) + h(n) nhỏ nhất.
    # g(n): chi phí thật từ start đến node hiện tại.
    # h(n): chi phí ước lượng từ node hiện tại đến goal.
    started_at = perf_counter()
    if not is_free(maze, start) or not is_free(maze, goal):
        return SearchResult(algorithm_name, False, [], None, 0, _elapsed_ms(started_at))

    # Heap ưu tiên luôn lấy node có f_score nhỏ nhất ra trước.
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
            # Mỗi bước đi trong mê cung có chi phí 1.
            tentative_g = g_score[current] + 1.0
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                counter += 1
                # Điểm quan trọng của A*: heuristic có thể là Manhattan hoặc ANN.
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
