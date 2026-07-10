"""Script so sánh hiệu năng tìm kiếm giữa BFS, A* Manhattan và A* ANN.

File này khởi tạo một mê cung ngẫu nhiên, chọn điểm Start và Goal, lần lượt áp dụng
các thuật toán tìm đường khác nhau, sau đó xuất báo cáo so sánh số liệu (thời gian, 
số node đã duyệt) ra terminal và lưu kết quả dạng biểu đồ PNG kèm file JSON.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import MODELS_DIR, OUTPUTS_DIR
from .dataset import generate_training_maze
from .heuristics import AnnHeuristic
from .maze import Position
from .search import astar, bfs_path, manhattan
from .visualize import save_search_figure


def parse_args() -> argparse.Namespace:
    """Phân tích các đối số dòng lệnh đầu vào cho script chạy demo.
    
    Returns:
        argparse.Namespace chứa các tham số dòng lệnh.
    """
    parser = argparse.ArgumentParser(description="So sánh hiệu năng BFS, A* Manhattan, và A* ANN.")
    parser.add_argument("--height", type=int, default=20, help="Chiều cao mê cung demo.")
    parser.add_argument("--width", type=int, default=20, help="Chiều rộng mê cung demo.")
    parser.add_argument("--wall-prob", type=float, default=0.25, help="Xác suất xuất hiện tường.")
    parser.add_argument("--min-goal-distance", type=int, default=20, help="Khoảng cách tối thiểu từ Start tới Goal.")
    parser.add_argument("--seed", type=int, default=123, help="Hạt giống ngẫu nhiên cho mê cung demo.")
    parser.add_argument(
        "--model",
        type=Path,
        default=MODELS_DIR / "ann_heuristic_goodfit.tflite",
        help="Đường dẫn đến file mô hình .tflite. Nếu không tìm thấy, thuật toán A* ANN sẽ bị bỏ qua.",
    )
    parser.add_argument("--output", type=Path, default=OUTPUTS_DIR / "demo_path.png", help="Đường dẫn lưu hình ảnh so sánh đường đi.")
    parser.add_argument("--metrics", type=Path, default=OUTPUTS_DIR / "demo_metrics.json", help="Đường dẫn lưu chỉ số so sánh dạng JSON.")
    return parser.parse_args()


def main() -> None:
    """Hàm chạy chính thực thi luồng so sánh demo giữa các thuật toán tìm đường."""
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # 1. Sinh mê cung ngẫu nhiên đảm bảo có đường đi tối thiểu
    maze, goal, distances = generate_training_maze(
        height=args.height,
        width=args.width,
        wall_probability=args.wall_prob,
        min_goal_distance=args.min_goal_distance,
        rng=rng,
    )
    # Chọn ngẫu nhiên vị trí Start từ tập các ô trống liên thông
    start = choose_demo_start(distances, min_distance=args.min_goal_distance, rng=rng)

    # 2. Chạy thử nghiệm các thuật toán cơ bản: BFS và A* Manhattan
    results = [
        bfs_path(maze, start, goal),
        astar(maze, start, goal, manhattan, algorithm_name="A* + Manhattan"),
    ]

    # 3. Nếu mô hình ANN đã được train và convert thành công, chạy A* ANN
    if args.model.exists():
        ann_heuristic = AnnHeuristic(maze=maze, model_path=args.model)
        results.append(
            astar(maze, start, goal, ann_heuristic, algorithm_name="A* + ANN")
        )
    else:
        print(f"Không tìm thấy mô hình, bỏ qua A* ANN: {args.model}")

    # 4. Vẽ biểu đồ so sánh đường đi và lưu ra file ảnh
    visual_output = save_search_figure(maze, start, goal, results, args.output)
    
    # 5. Lưu các chỉ số chi tiết ra file JSON
    save_metrics(results, args.metrics, start=start, goal=goal)

    # In kết quả dạng bảng tóm tắt ra màn hình terminal
    print(f"Vị trí bắt đầu (Start): {start}, Đích (Goal): {goal}")
    for result in results:
        status = "Tìm thấy" if result.found else "Không tìm thấy"
        print(
            f"{result.algorithm:16}: {status:14}, Độ dài={result.path_length}, "
            f"Số node duyệt={result.explored_nodes:3}, Thời gian={result.elapsed_ms:.2f} ms"
        )
    print(f"Đã lưu hình ảnh kết quả tìm kiếm tại: {visual_output}")
    print(f"Đã lưu các chỉ số thống kê tại: {args.metrics}")


def choose_demo_start(
    distances: np.ndarray,
    min_distance: int,
    rng: np.random.Generator,
) -> Position:
    """Chọn một vị trí xuất phát ngẫu nhiên có khoảng cách tối thiểu tới Goal đạt chuẩn.
    
    Args:
        distances: Ma trận khoảng cách BFS thực tế tới Goal.
        min_distance: Khoảng cách tối thiểu cần đạt.
        rng: Bộ sinh số ngẫu nhiên NumPy Generator.
        
    Returns:
        Tọa độ xuất phát (dòng, cột) được chọn.
    """
    # Lọc ra các ô trống thỏa mãn có đường đi tới đích và khoảng cách >= min_distance
    candidates = np.argwhere(np.isfinite(distances) & (distances >= min_distance))
    if len(candidates) == 0:
        candidates = np.argwhere(np.isfinite(distances))
    selected = candidates[int(rng.integers(0, len(candidates)))]
    return int(selected[0]), int(selected[1])


def save_metrics(
    results: list[SearchResult],
    output_path: Path,
    start: Position,
    goal: Position,
) -> None:
    """Lưu trữ số liệu hiệu năng của các thuật toán dưới định dạng JSON.
    
    Args:
        results: Danh sách kết quả SearchResult.
        output_path: Đường dẫn lưu file .json.
        start: Tọa độ điểm xuất phát.
        goal: Tọa độ điểm đích.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "start": [int(start[0]), int(start[1])],
        "goal": [int(goal[0]), int(goal[1])],
        "results": [
            {
                "algorithm": result.algorithm,
                "found": bool(result.found),
                "path_length": None if result.path_length is None else int(result.path_length),
                "explored_nodes": int(result.explored_nodes),
                "elapsed_ms": float(result.elapsed_ms),
            }
            for result in results
        ],
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()

