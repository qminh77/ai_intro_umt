"""Script dòng lệnh (CLI tool) để tự động sinh tập dữ liệu mê cung.

File này chứa hàm phân tích cú pháp dòng lệnh (arguments parser) và hàm main để thực hiện
sinh dữ liệu qua dataset.py rồi lưu thành file CSV.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import DATA_DIR
from .dataset import generate_labeled_dataset


def parse_args() -> argparse.Namespace:
    """Phân tích các đối số dòng lệnh đầu vào khi thực thi script.
    
    Returns:
        argparse.Namespace chứa giá trị các tham số đã phân tích cú pháp.
    """
    parser = argparse.ArgumentParser(description="Tự động sinh tập dữ liệu mê cung được dán nhãn bằng BFS.")
    parser.add_argument("--mazes", type=int, default=500, help="Số lượng mê cung cần sinh.")
    parser.add_argument("--height", type=int, default=20, help="Chiều cao mê cung.")
    parser.add_argument("--width", type=int, default=20, help="Chiều rộng mê cung.")
    parser.add_argument("--wall-prob", type=float, default=0.25, help="Xác suất tạo tường (0.0 -> 1.0).")
    parser.add_argument("--min-goal-distance", type=int, default=20, help="Khoảng cách tối thiểu từ Goal.")
    parser.add_argument("--max-samples-per-maze", type=int, default=200, help="Số mẫu tối đa lấy trên mỗi mê cung.")
    parser.add_argument("--seed", type=int, default=42, help="Hạt giống số ngẫu nhiên.")
    parser.add_argument("--output", type=Path, default=DATA_DIR / "maze_dataset.csv", help="Đường dẫn file CSV đầu ra.")
    return parser.parse_args()


def main() -> None:
    """Hàm chạy chính để khởi tạo thư mục và thực thi tiến trình sinh tập dữ liệu."""
    args = parse_args()
    # Tạo thư mục chứa dữ liệu nếu chưa tồn tại
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print(f"Bắt đầu sinh {args.mazes} mê cung...")
    df = generate_labeled_dataset(
        maze_count=args.mazes,
        height=args.height,
        width=args.width,
        wall_probability=args.wall_prob,
        min_goal_distance=args.min_goal_distance,
        max_samples_per_maze=args.max_samples_per_maze,
        seed=args.seed,
    )
    # Lưu kết quả DataFrame ra file CSV
    df.to_csv(args.output, index=False)

    print(f"Đã lưu tập dữ liệu tại: {args.output}")
    print(f"Số dòng: {len(df):,}")
    print(f"Các cột dữ liệu: {', '.join(df.columns)}")


if __name__ == "__main__":
    main()
