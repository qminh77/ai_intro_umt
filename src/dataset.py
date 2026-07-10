"""Sinh dataset huấn luyện ANN cho heuristic mê cung.

Ý tưởng chính: ta chưa có dữ liệu sẵn, nên tự tạo dữ liệu bằng cách sinh nhiều
mê cung ngẫu nhiên rồi dùng BFS tính khoảng cách thật tới goal. Khoảng cách đó
trở thành nhãn `true_distance` để ANN học dự đoán.
"""

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
    """Đổi một trạng thái mê cung thành vector 10 đặc trưng cho ANN."""

    row, col = current
    goal_row, goal_col = goal
    height, width = maze.shape

    # Chuẩn hóa tọa độ về khoảng 0..1 để mạng học ổn định hơn.
    # Nếu không chuẩn hóa, giá trị lớn nhỏ khác nhau có thể làm việc train khó hơn.
    max_x = max(width - 1, 1)
    max_y = max(height - 1, 1)

    return np.array(
        [
            col / max_x,                                   # x hiện tại.
            row / max_y,                                   # y hiện tại.
            goal_col / max_x,                              # x của đích.
            goal_row / max_y,                              # y của đích.
            abs(col - goal_col) / max_x,                   # khoảng cách ngang tới đích.
            abs(row - goal_row) / max_y,                   # khoảng cách dọc tới đích.
            float(is_wall_or_outside(maze, (row - 1, col))), # tường phía trên.
            float(is_wall_or_outside(maze, (row + 1, col))), # tường phía dưới.
            float(is_wall_or_outside(maze, (row, col - 1))), # tường bên trái.
            float(is_wall_or_outside(maze, (row, col + 1))), # tường bên phải.
        ],
        dtype=np.float32,
    )


def split_features_target(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    # Tách DataFrame thành X và y theo đúng cách gọi trong Machine Learning.
    # X là input 10 cột, y là nhãn khoảng cách thật.
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
    # rng có seed để kết quả có thể tái lập khi báo cáo/thuyết trình.
    rng = np.random.default_rng(seed)
    rows: list[dict[str, float]] = []

    for maze_id in range(maze_count):
        # Mỗi mê cung có một goal, BFS tính khoảng cách từ mọi ô tới goal đó.
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

    # Trộn dữ liệu để train/validation/test không bị gom theo từng mê cung liên tiếp.
    return pd.DataFrame(rows).sample(frac=1.0, random_state=seed).reset_index(drop=True)


def generate_training_maze(
    height: int,
    width: int,
    wall_probability: float,
    min_goal_distance: int,
    rng: np.random.Generator,
    max_attempts: int = 1000,
) -> tuple[np.ndarray, Position, np.ndarray]:
    # Không phải mê cung random nào cũng dùng được. Có map quá nhiều tường hoặc goal bị cô lập.
    # Vì vậy ta thử nhiều lần tới khi tìm được map có vùng đi được đủ lớn.
    for _ in range(max_attempts):
        maze = generate_maze(height, width, wall_probability, rng)
        if len(free_cells(maze)) < min_goal_distance:
            continue

        goal = random_free_cell(maze, rng)

        # distances là nhãn thô: mỗi ô có khoảng cách ngắn nhất tới goal.
        distances = bfs_distances_to_goal(maze, goal)
        finite_distances = distances[np.isfinite(distances)]

        # Chỉ nhận map nếu có đủ ô reachable và có ít nhất một ô xa goal.
        # Điều này giúp demo đường đi không quá ngắn và dataset có ý nghĩa hơn.
        if len(finite_distances) >= min_goal_distance and finite_distances.max() >= min_goal_distance:
            return maze, goal, distances

    raise RuntimeError(
        "Không sinh được mê cung hợp lệ. Hãy giảm wall_probability hoặc min_goal_distance."
    )


def build_dataset_rows(
    maze: np.ndarray,
    goal: Position,
    distances: np.ndarray,
    maze_id: int,
    max_samples: int | None,
    rng: np.random.Generator,
) -> list[dict[str, float]]:
    # Chỉ lấy các ô có khoảng cách hữu hạn, tức là có đường đi tới goal.
    reachable_positions = [
        (int(row), int(col))
        for row, col in zip(*np.where(np.isfinite(distances)), strict=True)
    ]

    # Nếu một mê cung có quá nhiều ô reachable, lấy mẫu bớt để dataset không quá lớn.
    if max_samples is not None and len(reachable_positions) > max_samples:
        selected = rng.choice(len(reachable_positions), size=max_samples, replace=False)
        reachable_positions = [reachable_positions[int(index)] for index in selected]

    rows: list[dict[str, float]] = []
    for position in reachable_positions:
        # Mỗi position trở thành một dòng dữ liệu: 10 feature + maze_id + true_distance.
        values = state_to_features(maze, position, goal)
        row = {name: float(value) for name, value in zip(FEATURE_COLUMNS, values, strict=True)}
        row["maze_id"] = float(maze_id)
        row[TARGET_COLUMN] = float(distances[position])
        rows.append(row)
    return rows


def parse_args() -> argparse.Namespace:
    # Các tham số này cho phép tự thay đổi kích thước mê cung và số lượng dữ liệu.
    parser = argparse.ArgumentParser(description="Sinh dataset mê cung được gán nhãn bằng BFS.")
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

    print(f"Sinh {args.mazes} mê cung...")
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
    print(f"Đã lưu dataset: {args.output}")
    print(f"Số dòng: {len(df):,}")


if __name__ == "__main__":
    main()
