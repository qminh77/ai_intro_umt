"""Trích xuất và chuẩn bị các đặc trưng đầu vào cho mạng ANN từ trạng thái mê cung.

File này chứa các hàm chuyển đổi thông tin tọa độ tác tử, điểm đích và cảm biến tường lân cận
thành vector đặc trưng 10 chiều được chuẩn hóa (normalised), hỗ trợ gom cụm dòng dữ liệu cho CSV
và chia tách ma trận đặc trưng X / nhãn mục tiêu Y phục vụ huấn luyện.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .config import FEATURE_COLUMNS, TARGET_COLUMN
from .maze import Position, is_wall_or_outside


def state_to_features(maze: np.ndarray, current: Position, goal: Position) -> np.ndarray:
    """Chuyển đổi một trạng thái trong mê cung thành vector 10 đặc trưng đầu vào chuẩn hóa cho mạng ANN.
    
    10 đặc trưng bao gồm:
    - 2 chiều tọa độ hiện tại chuẩn hóa (x, y)
    - 2 chiều tọa độ đích chuẩn hóa (goal_x, goal_y)
    - 2 chiều khoảng cách tuyệt đối chuẩn hóa (dx, dy)
    - 4 chiều trạng thái tường xung quanh (lên, xuống, trái, phải) dạng nhị phân 0 hoặc 1.
    
    Args:
        maze: Ma trận mê cung 2D.
        current: Tọa độ hiện tại của tác tử (dòng, cột).
        goal: Tọa độ đích đến (dòng, cột).
        
    Returns:
        Mảng NumPy 1D kiểu np.float32 có kích thước (10,).
    """
    row, col = current
    goal_row, goal_col = goal
    height, width = maze.shape

    # Tránh chia cho 0 nếu kích thước mê cung là 1x1
    max_x = max(width - 1, 1)
    max_y = max(height - 1, 1)

    dx = abs(col - goal_col)
    dy = abs(row - goal_row)

    values = [
        col / max_x,           # x chuẩn hóa (0.0 -> 1.0)
        row / max_y,           # y chuẩn hóa (0.0 -> 1.0)
        goal_col / max_x,      # goal_x chuẩn hóa (0.0 -> 1.0)
        goal_row / max_y,      # goal_y chuẩn hóa (0.0 -> 1.0)
        dx / max_x,            # dx chuẩn hóa (0.0 -> 1.0)
        dy / max_y,            # dy chuẩn hóa (0.0 -> 1.0)
        float(is_wall_or_outside(maze, (row - 1, col))),   # Cảm biến tường Phía Trên (0: trống, 1: tường/ngoài biên)
        float(is_wall_or_outside(maze, (row + 1, col))),   # Cảm biến tường Phía Dưới (0: trống, 1: tường/ngoài biên)
        float(is_wall_or_outside(maze, (row, col - 1))),   # Cảm biến tường Phía Trái (0: trống, 1: tường/ngoài biên)
        float(is_wall_or_outside(maze, (row, col + 1))),   # Cảm biến tường Phía Phải (0: trống, 1: tường/ngoài biên)
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
    """Xây dựng danh sách các dòng đặc trưng từ một mê cung đã được BFS gán nhãn khoảng cách.
    
    Hàm này duyệt qua các ô có thể đi tới Goal, trích xuất đặc trưng và lưu thông tin
    maze_id kèm theo khoảng cách thực tế (true_distance) làm nhãn huấn luyện.
    
    Args:
        maze: Ma trận mê cung 2D.
        goal: Tọa độ đích đến.
        distances: Ma trận khoảng cách thực tế từ mọi ô đến goal (thu được từ BFS).
        maze_id: ID định danh của mê cung hiện tại.
        max_samples: Số lượng mẫu dữ liệu tối đa cần lấy ra từ mê cung này (nếu None thì lấy toàn bộ).
        rng: Bộ sinh số ngẫu nhiên NumPy Generator.
        
    Returns:
        Danh sách các dictionary, mỗi dict chứa các cặp tên cột đặc trưng và giá trị của nó.
    """
    # Tìm tất cả các tọa độ ô trống có thể đi đến đích (khoảng cách hữu hạn, khác vô cùng)
    reachable_positions = [
        (int(row), int(col))
        for row, col in zip(*np.where(np.isfinite(distances)), strict=True)
    ]

    # Nếu có giới hạn mẫu, thực hiện lấy mẫu ngẫu nhiên không lặp
    if max_samples is not None and len(reachable_positions) > max_samples:
        selected = rng.choice(len(reachable_positions), size=max_samples, replace=False)
        reachable_positions = [reachable_positions[int(index)] for index in selected]

    rows: list[dict[str, float]] = []
    for position in reachable_positions:
        # Chuyển đổi trạng thái hiện tại thành vector 10 đặc trưng
        features = state_to_features(maze, position, goal)
        # Đóng gói dữ liệu đặc trưng vào dict
        row = {name: float(value) for name, value in zip(FEATURE_COLUMNS, features, strict=True)}
        row["maze_id"] = float(maze_id)
        row[TARGET_COLUMN] = float(distances[position])  # Gán nhãn đích (true_distance)
        rows.append(row)

    return rows


def split_features_target(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Tách biệt DataFrame chứa tập dữ liệu thành ma trận đầu vào X và mảng nhãn đích Y.
    
    Args:
        df: Pandas DataFrame chứa tập dữ liệu mê cung.
        
    Returns:
        Một tuple (X, Y) trong đó:
        - X là mảng NumPy 2D chứa 10 đặc trưng đầu vào.
        - Y là mảng NumPy 1D chứa khoảng cách thực tế làm nhãn.
    """
    x = df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y = df[TARGET_COLUMN].to_numpy(dtype=np.float32)
    return x, y


