"""Demo so sánh BFS, A* Manhattan và A* dùng ANN heuristic.

File này là phần nâng cao của đề tài: thay vì tự viết heuristic bằng tay, ta
load model ANN đã train và dùng dự đoán của model làm h(n) trong A*.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import MODELS_DIR, OUTPUTS_DIR
from .dataset import generate_training_maze, state_to_features
from .maze import Position, SearchResult, WALL, astar, bfs_path, manhattan
from .train import import_tensorflow


class AnnHeuristic:
    """Biến model Keras thành hàm h(n) để truyền vào A*."""

    def __init__(self, maze: np.ndarray, model_path: Path) -> None:
        # Load model goodfit đã train ở bước `python -m src.train --experiment all`.
        tf = import_tensorflow()
        self.maze = maze
        self.model = tf.keras.models.load_model(model_path)

        # Cache giúp cùng một trạng thái không phải gọi model.predict nhiều lần.
        self.cache: dict[tuple[Position, Position], float] = {}

    def __call__(self, position: Position, goal: Position) -> float:
        # A* sẽ gọi object này như một hàm: heuristic(position, goal).
        key = (position, goal)
        if key not in self.cache:
            # Chuyển trạng thái mê cung thành đúng 10 feature như lúc train.
            features = state_to_features(self.maze, position, goal).reshape(1, -1)

            # Model dự đoán khoảng cách còn lại tới goal.
            prediction = float(self.model.predict(features, verbose=0)[0][0])

            # Heuristic không nên âm, nên ép giá trị nhỏ nhất là 0.
            self.cache[key] = max(0.0, prediction)
        return self.cache[key]


def choose_demo_start(
    distances: np.ndarray,
    min_distance: int,
    rng: np.random.Generator,
) -> Position:
    # Chọn start đủ xa goal để demo đường đi có ý nghĩa, không quá ngắn.
    candidates = np.argwhere(np.isfinite(distances) & (distances >= min_distance))
    if len(candidates) == 0:
        # Nếu không có ô đủ xa, lấy bất kỳ ô nào đi tới được goal.
        candidates = np.argwhere(np.isfinite(distances))
    selected = candidates[int(rng.integers(0, len(candidates)))]
    return int(selected[0]), int(selected[1])


def save_search_figure(
    maze: np.ndarray,
    start: Position,
    goal: Position,
    results: list[SearchResult],
    output_path: Path,
) -> Path:
    # Vẽ 1 hình gồm nhiều subplot: BFS, A* Manhattan, A* ANN.
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError:
        fallback_path = output_path.with_suffix(".txt")
        save_ascii_maze(maze, start, goal, results, fallback_path)
        return fallback_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5.7))
    if len(results) == 1:
        axes = [axes]

    cmap = ListedColormap(["white", "black"])
    for axis, result in zip(axes, results, strict=True):
        # Nền mê cung: ô trắng là đường đi, ô đen là tường.
        axis.imshow(maze == WALL, cmap=cmap, origin="upper")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(_title_for_result(result), fontsize=11, pad=10)

        if result.path:
            # Đường màu đỏ là path cuối cùng thuật toán tìm được.
            rows = [position[0] for position in result.path]
            cols = [position[1] for position in result.path]
            axis.plot(cols, rows, color="#ef4444", linewidth=2.5)

        axis.scatter(start[1], start[0], c="#22c55e", s=110, label="Start", marker="o")
        axis.scatter(goal[1], goal[0], c="#3b82f6", s=130, label="Goal", marker="*")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=True)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.14, wspace=0.08)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path


def save_ascii_maze(
    maze: np.ndarray,
    start: Position,
    goal: Position,
    results: list[SearchResult],
    output_path: Path,
) -> None:
    # Fallback khi môi trường không có matplotlib: lưu mê cung bằng ký tự ASCII.
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["Matplotlib chưa được cài, lưu mê cung dạng ASCII.", ""]

    for result in results:
        path_cells = set(result.path)
        lines.append(_title_for_result(result))
        for row in range(maze.shape[0]):
            chars = []
            for col in range(maze.shape[1]):
                position = (row, col)
                if position == start:
                    chars.append("S")
                elif position == goal:
                    chars.append("G")
                elif maze[position] == WALL:
                    chars.append("#")
                elif position in path_cells:
                    chars.append("*")
                else:
                    chars.append(".")
            lines.append("".join(chars))
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def save_metrics(
    results: list[SearchResult],
    output_path: Path,
    start: Position,
    goal: Position,
) -> None:
    # Lưu số liệu để đưa vào báo cáo: độ dài path, số node duyệt, thời gian chạy.
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


def _title_for_result(result: SearchResult) -> str:
    if not result.found:
        return f"{result.algorithm}: không có đường, duyệt={result.explored_nodes}"
    return (
        f"{result.algorithm}: độ dài={result.path_length}, "
        f"duyệt={result.explored_nodes}, {result.elapsed_ms:.2f} ms"
    )


def parse_args() -> argparse.Namespace:
    # Cho phép đổi kích thước map, seed và đường dẫn model khi chạy demo.
    parser = argparse.ArgumentParser(description="So sánh BFS, A* Manhattan và A* ANN.")
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--wall-prob", type=float, default=0.25)
    parser.add_argument("--min-goal-distance", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--model", type=Path, default=MODELS_DIR / "ann_heuristic_goodfit.keras")
    parser.add_argument("--output", type=Path, default=OUTPUTS_DIR / "demo_path.png")
    parser.add_argument("--metrics", type=Path, default=OUTPUTS_DIR / "demo_metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    # Sinh một mê cung mới cho demo và lấy luôn bảng khoảng cách BFS tới goal.
    maze, goal, distances = generate_training_maze(
        height=args.height,
        width=args.width,
        wall_probability=args.wall_prob,
        min_goal_distance=args.min_goal_distance,
        rng=rng,
    )
    start = choose_demo_start(distances, min_distance=args.min_goal_distance, rng=rng)

    # Hai baseline truyền thống: BFS không heuristic và A* dùng Manhattan.
    results = [
        bfs_path(maze, start, goal),
        astar(maze, start, goal, manhattan, algorithm_name="A* + Manhattan"),
    ]

    # Nếu model goodfit tồn tại, thêm A* ANN vào bảng so sánh.
    if args.model.exists():
        ann_heuristic = AnnHeuristic(maze=maze, model_path=args.model)
        results.append(astar(maze, start, goal, ann_heuristic, algorithm_name="A* + ANN"))
    else:
        print(f"Không tìm thấy model, bỏ qua A* ANN: {args.model}")

    # Lưu hình và metrics để dùng trong slide/báo cáo.
    visual_output = save_search_figure(maze, start, goal, results, args.output)
    save_metrics(results, args.metrics, start=start, goal=goal)

    print(f"Start: {start}, Goal: {goal}")
    for result in results:
        status = "Tìm thấy" if result.found else "Không tìm thấy"
        print(
            f"{result.algorithm:16}: {status:14}, độ dài={result.path_length}, "
            f"node duyệt={result.explored_nodes:3}, thời gian={result.elapsed_ms:.2f} ms"
        )
    print(f"Đã lưu hình: {visual_output}")
    print(f"Đã lưu metrics: {args.metrics}")


if __name__ == "__main__":
    main()
