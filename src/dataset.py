"""Logic xây dựng và tổng hợp tập dữ liệu từ các mê cung ngẫu nhiên được gán nhãn.

File này chịu trách nhiệm điều phối việc sinh ra nhiều mê cung ngẫu nhiên thỏa mãn điều kiện 
khoảng cách tối thiểu tới Goal, chạy giải thuật BFS từ Goal để đánh nhãn khoảng cách thực tế,
và tổng hợp tất cả các dòng dữ liệu thành một Pandas DataFrame được trộn ngẫu nhiên.
"""

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
    """Sinh ra tập dữ liệu mê cung có gắn nhãn hoàn chỉnh.
    
    Sinh nhiều mê cung, chọn đích Goal hợp lệ, chạy BFS để tính khoảng cách thực tế từ mọi ô
    đến Goal, trích xuất đặc trưng và gom tất cả thành một DataFrame được trộn ngẫu nhiên.
    
    Args:
        maze_count: Số lượng mê cung cần tạo.
        height: Chiều cao (số dòng) của mỗi mê cung.
        width: Chiều rộng (số cột) của mỗi mê cung.
        wall_probability: Xác suất tạo tường trong mỗi ô.
        min_goal_distance: Khoảng cách ngắn nhất tối thiểu từ ô xa nhất đến Goal.
        max_samples_per_maze: Số mẫu dữ liệu tối đa trích xuất từ mỗi mê cung.
        seed: Hạt giống ngẫu nhiên để tái lập kết quả sinh dữ liệu.
        
    Returns:
        Pandas DataFrame được trộn ngẫu nhiên chứa toàn bộ tập dữ liệu huấn luyện/kiểm thử.
    """
    rng = np.random.default_rng(seed)
    all_rows: list[dict[str, float]] = []

    for maze_id in range(maze_count):
        # Tạo mê cung thỏa mãn điều kiện và tính khoảng cách từ Goal
        maze, goal, distances = generate_training_maze(
            height=height,
            width=width,
            wall_probability=wall_probability,
            min_goal_distance=min_goal_distance,
            rng=rng,
        )
        # Trích xuất các dòng đặc trưng và thêm vào danh sách tổng hợp
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

    # Chuyển đổi thành DataFrame và trộn ngẫu nhiên toàn bộ dữ liệu
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
    """Sinh ra một mê cung ngẫu nhiên hợp lệ và có đủ số lượng ô trống huấn luyện.
    
    Mê cung hợp lệ phải có Goal nằm ở ô trống và khoảng cách từ ô trống xa nhất trong mê cung 
    tới Goal phải đạt ít nhất bằng `min_goal_distance`.
    
    Args:
        height: Chiều cao mê cung.
        width: Chiều rộng mê cung.
        wall_probability: Xác suất tạo tường.
        min_goal_distance: Khoảng cách thực tế tối thiểu giữa Goal và ô xa nhất.
        rng: Bộ sinh số ngẫu nhiên NumPy Generator.
        max_attempts: Số lần thử tạo lại mê cung tối đa trước khi báo lỗi.
        
    Returns:
        Một Tuple (maze, goal, distances) trong đó:
        - maze: Ma trận mê cung 2D.
        - goal: Tọa độ đích đến ngẫu nhiên được chọn.
        - distances: Ma trận khoảng cách BFS từ các ô trống tới goal.
        
    Raises:
        RuntimeError: Nếu vượt quá `max_attempts` lần thử mà không sinh được mê cung hợp lệ.
    """
    for _ in range(max_attempts):
        maze = generate_maze(height, width, wall_probability, rng)
        cells = free_cells(maze)
        
        # Bỏ qua nếu mê cung có quá ít ô trống (không đủ để đạt min_goal_distance)
        if len(cells) < min_goal_distance:
            continue

        goal = random_free_cell(maze, rng)
        distances = bfs_distances_to_goal(maze, goal)
        finite_distances = distances[np.isfinite(distances)]

        # Mê cung phải có ít nhất `min_goal_distance` ô trống liên thông được với Goal
        if len(finite_distances) < min_goal_distance:
            continue
        # Khoảng cách tối đa từ Goal tới một ô trống bất kỳ trong tập liên thông phải đạt tối thiểu `min_goal_distance`
        if finite_distances.max() < min_goal_distance:
            continue

        return maze, goal, distances

    raise RuntimeError(
        "Không thể sinh được mê cung phù hợp sau nhiều lần thử. Hãy giảm bớt wall_probability hoặc min_goal_distance."
    )


