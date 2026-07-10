"""Triển khai các thuật toán tìm kiếm đường đi và khoảng cách trên mê cung.

File này chứa mã nguồn cho thuật toán BFS (Tìm kiếm theo chiều rộng) để tính khoảng cách thực tế 
làm nhãn và tìm đường đi mù, thuật toán A* hỗ trợ hàm Heuristic tổng quát, và các hàm bổ trợ khác.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from heapq import heappop, heappush
from time import perf_counter
from typing import Callable

import numpy as np

from .maze import Position, iter_neighbors, is_free

# Định nghĩa kiểu dữ liệu cho hàm Heuristic (nhận vị trí hiện tại và đích, trả về khoảng cách ước lượng)
Heuristic = Callable[[Position, Position], float]


@dataclass(frozen=True)
class SearchResult:
    """Lớp chứa kết quả sau khi thực hiện thuật toán tìm kiếm đường đi.
    
    Attributes:
        algorithm: Tên của thuật toán đã sử dụng (ví dụ: BFS, A* Manhattan, A* ANN).
        found: Trạng thái tìm thấy đường đi hay không (True/False).
        path: Danh sách các tọa độ Position tạo nên đường đi từ Start đến Goal.
        path_length: Độ dài của đường đi tìm được (bằng số bước di chuyển, hoặc None nếu không thấy).
        explored_nodes: Tổng số lượng node (ô) đã được duyệt/mở rộng trong quá trình tìm kiếm.
        elapsed_ms: Thời gian thực thi thuật toán tính bằng mili-giây.
        explored_order: Thứ tự duyệt qua các node, dùng để hiển thị animation trên UI.
    """
    algorithm: str
    found: bool
    path: list[Position]
    path_length: int | None
    explored_nodes: int
    elapsed_ms: float
    explored_order: list[Position] = field(default_factory=list)


def manhattan(position: Position, goal: Position) -> float:
    """Tính khoảng cách Manhattan giữa hai vị trí trên lưới.
    
    Khoảng cách Manhattan được tính bằng tổng độ lệch tuyệt đối theo trục ngang và trục dọc.
    
    Args:
        position: Tọa độ điểm hiện tại.
        goal: Tọa độ điểm đích.
        
    Returns:
        Giá trị khoảng cách Manhattan (số thực).
    """
    return abs(position[0] - goal[0]) + abs(position[1] - goal[1])


def reconstruct_path(
    came_from: dict[Position, Position],
    start: Position,
    goal: Position,
) -> list[Position]:
    """Tái dựng lại đường đi từ điểm xuất phát đến điểm đích dựa trên bảng vết truy vết.
    
    Args:
        came_from: Từ điển lưu vết node cha của mỗi node (Node hiện tại -> Node cha).
        start: Tọa độ xuất phát.
        goal: Tọa độ đích.
        
    Returns:
        Danh sách tọa độ đi từ start đến goal. Trả về danh sách rỗng nếu không có đường đi.
    """
    if goal == start:
        return [start]
    if goal not in came_from:
        return []

    current = goal
    path = [current]
    # Truy vết ngược từ đích về điểm xuất phát
    while current != start:
        current = came_from[current]
        path.append(current)
    path.reverse()  # Đảo ngược danh sách để có đường đi đúng chiều từ Start -> Goal
    return path


def bfs_distances_to_goal(maze: np.ndarray, goal: Position) -> np.ndarray:
    """Tính khoảng cách ngắn nhất thực tế từ mọi ô trống có thể đi đến Goal bằng thuật toán BFS.
    
    Hàm này chạy BFS từ Goal lan ra toàn bộ mê cung, giúp thu thập nhãn khoảng cách chính xác 
    (`true_distance`) cho toàn bộ các ô trong mê cung phục vụ huấn luyện ANN.
    
    Args:
        maze: Ma trận mê cung 2D.
        goal: Tọa độ điểm đích (Goal).
        
    Returns:
        Ma trận 2D chứa khoảng cách thực tế từ từng ô đến Goal. 
        Những ô không thể đi tới được (bị cô lập bởi tường) sẽ có giá trị np.inf.
        
    Raises:
        ValueError: Nếu điểm đích nằm trên ô tường (không hợp lệ).
    """
    if not is_free(maze, goal):
        raise ValueError("Điểm đích phải là một ô trống (không phải tường).")

    # Khởi tạo ma trận khoảng cách với giá trị vô cùng (np.inf)
    distances = np.full(maze.shape, np.inf, dtype=float)
    distances[goal] = 0.0
    queue: deque[Position] = deque([goal])

    while queue:
        current = queue.popleft()
        for neighbor in iter_neighbors(maze, current):
            # Nếu ô lân cận chưa được cập nhật khoảng cách (vẫn là vô cùng)
            if np.isinf(distances[neighbor]):
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)

    return distances


def bfs_path(maze: np.ndarray, start: Position, goal: Position) -> SearchResult:
    """Tìm đường đi ngắn nhất từ Start đến Goal bằng thuật toán BFS (Tìm kiếm mù).
    
    BFS đảm bảo tìm được đường đi ngắn nhất trên đồ thị không trọng số, duyệt qua các node
    theo từng tầng khoảng cách từ điểm xuất phát.
    
    Args:
        maze: Ma trận mê cung 2D.
        start: Tọa độ xuất phát.
        goal: Tọa độ đích.
        
    Returns:
        Đối tượng SearchResult chứa kết quả tìm kiếm của BFS.
    """
    started_at = perf_counter()
    # Kiểm tra tính hợp lệ của điểm xuất phát và điểm đích
    if not is_free(maze, start) or not is_free(maze, goal):
        return SearchResult("BFS", False, [], None, 0, _elapsed_ms(started_at))

    queue: deque[Position] = deque([start])
    came_from: dict[Position, Position] = {}
    visited = {start}
    explored_nodes = 0
    explored_order: list[Position] = []

    while queue:
        current = queue.popleft()
        explored_nodes += 1
        explored_order.append(current)
        
        # Nếu đã chạm tới đích, kết thúc và tái dựng đường đi
        if current == goal:
            path = reconstruct_path(came_from, start, goal)
            return SearchResult(
                "BFS",
                True,
                path,
                len(path) - 1,
                explored_nodes,
                _elapsed_ms(started_at),
                explored_order,
            )

        for neighbor in iter_neighbors(maze, current):
            if neighbor not in visited:
                visited.add(neighbor)
                came_from[neighbor] = current
                queue.append(neighbor)

    # Không tìm thấy đường đi tới đích
    return SearchResult(
        "BFS", False, [], None, explored_nodes, _elapsed_ms(started_at), explored_order
    )


def astar(
    maze: np.ndarray,
    start: Position,
    goal: Position,
    heuristic: Heuristic,
    algorithm_name: str = "A*",
) -> SearchResult:
    """Thuật toán tìm kiếm A* tổng quát.
    
    Sử dụng hàm đánh giá f(n) = g(n) + h(n) để định hướng tìm kiếm.
    Hỗ trợ truyền vào hàm heuristic tùy ý (ví dụ: Manhattan Heuristic hoặc ANN Heuristic).
    
    Args:
        maze: Ma trận mê cung 2D.
        start: Tọa độ xuất phát.
        goal: Tọa độ đích.
        heuristic: Hàm tính toán heuristic h(n).
        algorithm_name: Tên hiển thị của thuật toán phục vụ so sánh.
        
    Returns:
        Đối tượng SearchResult chứa kết quả tìm kiếm của A*.
    """
    started_at = perf_counter()
    if not is_free(maze, start) or not is_free(maze, goal):
        return SearchResult(algorithm_name, False, [], None, 0, _elapsed_ms(started_at))

    # Hàng đợi ưu tiên (Min-Heap): chứa các phần tử dạng (f_score, tie_breaker_counter, position)
    open_heap: list[tuple[float, int, Position]] = []
    heappush(open_heap, (heuristic(start, goal), 0, start))

    came_from: dict[Position, Position] = {}
    g_score: dict[Position, float] = {start: 0.0}
    closed: set[Position] = set()
    counter = 0  # Dùng để phân định thứ tự ưu tiên khi hai node có cùng f_score (tie breaker)
    explored_nodes = 0
    explored_order: list[Position] = []

    while open_heap:
        _, _, current = heappop(open_heap)
        
        # Bỏ qua nếu node này đã nằm trong tập đóng (đã duyệt qua rồi)
        if current in closed:
            continue

        closed.add(current)
        explored_nodes += 1
        explored_order.append(current)

        # Đã tìm thấy đích
        if current == goal:
            path = reconstruct_path(came_from, start, goal)
            return SearchResult(
                algorithm_name,
                True,
                path,
                len(path) - 1,
                explored_nodes,
                _elapsed_ms(started_at),
                explored_order,
            )

        for neighbor in iter_neighbors(maze, current):
            # Chi phí g thực tế để đi từ start qua current đến neighbor (mỗi bước đi có trọng số là 1.0)
            tentative_g = g_score[current] + 1.0
            
            # Nếu tìm thấy một đường đi ngắn hơn đến neighbor so với ghi nhận trước đó
            if tentative_g < g_score.get(neighbor, float("inf")):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score = tentative_g + float(heuristic(neighbor, goal))
                counter += 1
                heappush(open_heap, (f_score, counter, neighbor))

    # Không tìm thấy đường đi tới đích
    return SearchResult(
        algorithm_name,
        False,
        [],
        None,
        explored_nodes,
        _elapsed_ms(started_at),
        explored_order,
    )


def _elapsed_ms(started_at: float) -> float:
    """Tính toán thời gian trôi qua kể từ thời điểm `started_at` theo đơn vị mili-giây.
    
    Args:
        started_at: Thời điểm bắt đầu ghi nhận bằng hàm perf_counter().
        
    Returns:
        Thời gian trôi qua dưới dạng số thực (ms).
    """
    return (perf_counter() - started_at) * 1000

