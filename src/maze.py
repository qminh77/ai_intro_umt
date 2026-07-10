"""Các hàm xử lý cấu trúc lưới mê cung.

File này định nghĩa cách khởi tạo lưới mê cung ngẫu nhiên và cung cấp các hàm tiện ích
để kiểm tra biên, ô trống, ô tường, và duyệt qua các ô lân cận hợp lệ.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

# Định nghĩa các trạng thái của ô trong mê cung
FREE = 0  # Ô trống (có thể đi qua)
WALL = 1  # Ô tường (vật cản, không thể đi qua)

# Định nghĩa kiểu dữ liệu cho tọa độ (dòng, cột)
Position = tuple[int, int]

# 4 hướng di chuyển cơ bản trên lưới 2D (Lên, Xuống, Trái, Phải)
DIRECTIONS: tuple[Position, ...] = (
    (-1, 0),  # Lên (up)
    (1, 0),   # Xuống (down)
    (0, -1),  # Trái (left)
    (0, 1),   # Phải (right)
)


def generate_maze(
    height: int,
    width: int,
    wall_probability: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Tạo ngẫu nhiên một lưới mê cung 2D dạng ma trận nhị phân.
    
    Các ô có giá trị 0 là ô trống (đường đi) và 1 là tường (vật cản).
    
    Args:
        height: Chiều cao (số dòng) của mê cung.
        width: Chiều rộng (số cột) của mê cung.
        wall_probability: Xác suất một ô bất kỳ bị biến thành tường.
        rng: Bộ sinh số ngẫu nhiên NumPy Generator để kiểm soát seed.
        
    Returns:
        Ma trận NumPy 2D kiểu np.int8 biểu diễn mê cung.
    """
    maze = (rng.random((height, width)) < wall_probability).astype(np.int8)
    return maze


def in_bounds(maze: np.ndarray, position: Position) -> bool:
    """Kiểm tra một tọa độ có nằm trong phạm vi kích thước của mê cung hay không.
    
    Args:
        maze: Ma trận mê cung 2D.
        position: Tọa độ cần kiểm tra (dòng, cột).
        
    Returns:
        True nếu tọa độ nằm trong biên, ngược lại False.
    """
    row, col = position
    return 0 <= row < maze.shape[0] and 0 <= col < maze.shape[1]


def is_free(maze: np.ndarray, position: Position) -> bool:
    """Kiểm tra một ô có phải là ô trống (có thể đi qua) hay không.
    
    Args:
        maze: Ma trận mê cung 2D.
        position: Tọa độ cần kiểm tra (dòng, cột).
        
    Returns:
        True nếu ô nằm trong biên và có giá trị bằng FREE (0), ngược lại False.
    """
    return in_bounds(maze, position) and maze[position] == FREE


def is_wall_or_outside(maze: np.ndarray, position: Position) -> bool:
    """Kiểm tra một ô có phải là tường hoặc nằm ngoài biên mê cung hay không.
    
    Hàm này cực kỳ hữu ích cho việc tính toán các cảm biến tường lân cận của mô hình ANN.
    
    Args:
        maze: Ma trận mê cung 2D.
        position: Tọa độ cần kiểm tra (dòng, cột).
        
    Returns:
        True nếu ô nằm ngoài biên hoặc là tường (WALL = 1), ngược lại False.
    """
    return not in_bounds(maze, position) or maze[position] == WALL


def free_cells(maze: np.ndarray) -> list[Position]:
    """Lấy danh sách tất cả các tọa độ ô trống trong mê cung.
    
    Args:
        maze: Ma trận mê cung 2D.
        
    Returns:
        Danh sách các Tuple (dòng, cột) chứa tọa độ các ô trống.
    """
    rows, cols = np.where(maze == FREE)
    return list(zip(rows.astype(int), cols.astype(int), strict=True))


def random_free_cell(maze: np.ndarray, rng: np.random.Generator) -> Position:
    """Chọn ngẫu nhiên một ô trống trong mê cung.
    
    Thường dùng để khởi tạo ngẫu nhiên vị trí Start hoặc Goal cho tác tử.
    
    Args:
        maze: Ma trận mê cung 2D.
        rng: Bộ sinh số ngẫu nhiên NumPy Generator.
        
    Returns:
        Tọa độ (dòng, cột) của ô trống ngẫu nhiên được chọn.
        
    Raises:
        ValueError: Nếu mê cung không có ô trống nào.
    """
    cells = free_cells(maze)
    if not cells:
        raise ValueError("Mê cung không có ô trống nào.")
    row, col = cells[int(rng.integers(0, len(cells)))]
    return int(row), int(col)


def iter_neighbors(maze: np.ndarray, position: Position) -> Iterable[Position]:
    """Duyệt qua các ô trống lân cận hợp lệ của một vị trí theo 4 hướng di chuyển.
    
    Args:
        maze: Ma trận mê cung 2D.
        position: Tọa độ gốc (dòng, cột).
        
    Yields:
        Tọa độ các ô lân cận trống và nằm trong biên.
    """
    row, col = position
    for d_row, d_col in DIRECTIONS:
        candidate = (row + d_row, col + d_col)
        if is_free(maze, candidate):
            yield candidate

