"""Cung cấp các hàm vẽ trực quan hóa mê cung và kết quả tìm đường.

File này chứa các hàm dựng hình ảnh trực quan so sánh đường đi của các thuật toán 
bằng Matplotlib (lưu thành file ảnh PNG) hoặc tự động chuyển sang chế độ ASCII 
(lưu thành file văn bản TXT) nếu thư viện Matplotlib không khả dụng.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .maze import Position, WALL
from .search import SearchResult


def save_search_figure(
    maze: np.ndarray,
    start: Position,
    goal: Position,
    results: list[SearchResult],
    output_path: Path,
) -> Path:
    """Vẽ và lưu trữ biểu đồ trực quan hóa so sánh đường đi giữa các giải thuật.
    
    Hàm tạo ra các trục tọa độ song song (subplots), mỗi trục hiển thị một kết quả tìm đường.
    Nếu không có thư viện Matplotlib, hàm tự động chuyển sang chế độ fallback lưu dạng ASCII.
    
    Args:
        maze: Ma trận mê cung 2D.
        start: Tọa độ điểm bắt đầu.
        goal: Tọa độ điểm đích.
        results: Danh sách kết quả SearchResult từ các thuật toán.
        output_path: Đường dẫn lưu ảnh PNG.
        
    Returns:
        Đường dẫn thực tế đã lưu file (có thể là .png hoặc .txt).
    """
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError:
        # Nếu thiết bị chạy không hỗ trợ đồ họa hoặc thiếu Matplotlib, sinh file ASCII văn bản
        fallback_path = output_path.with_suffix(".txt")
        save_ascii_maze(maze, start, goal, results, fallback_path)
        return fallback_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    # Khởi tạo khung hình phụ dựa trên số lượng thuật toán cần vẽ
    fig, axes = plt.subplots(1, len(results), figsize=(5 * len(results), 5.7))
    if len(results) == 1:
        axes = [axes]

    # Map màu: ô trống vẽ màu trắng, ô tường vẽ màu đen
    cmap = ListedColormap(["white", "black"])

    for axis, result in zip(axes, results, strict=True):
        # Hiển thị nền mê cung
        axis.imshow(maze == WALL, cmap=cmap, origin="upper")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(_title_for_result(result), fontsize=11, pad=10)

        # Vẽ đường đi màu đỏ nối các tọa độ trong kết quả tìm được
        if result.path:
            rows = [position[0] for position in result.path]
            cols = [position[1] for position in result.path]
            axis.plot(cols, rows, color="#ef4444", linewidth=2.5)

        # Đánh dấu điểm Start (hình tròn xanh lá) và Goal (ngôi sao xanh dương)
        axis.scatter(start[1], start[0], c="#22c55e", s=110, label="Start", marker="o")
        axis.scatter(goal[1], goal[0], c="#3b82f6", s=130, label="Goal", marker="*")

    # Hiển thị chú thích (legend) chung phía dưới cùng
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
    """Sinh ảnh mê cung và đường đi biểu diễn bằng ký tự văn bản ASCII lưu ra file .txt.
    
    Ký tự biểu diễn:
    - 'S': Điểm xuất phát (Start).
    - 'G': Điểm đích (Goal).
    - '#': Ô tường (Wall).
    - '*': Ô nằm trên đường đi tìm được.
    - '.': Ô trống bình thường.
    
    Args:
        maze: Ma trận mê cung 2D.
        start: Tọa độ xuất phát.
        goal: Tọa độ đích.
        results: Danh sách kết quả tìm kiếm.
        output_path: Đường dẫn file văn bản đầu ra.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Thư viện Matplotlib chưa được cài đặt, tự động tạo mê cung dạng ký tự ASCII.",
        "",
    ]

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


def _title_for_result(result: SearchResult) -> str:
    """Tạo tiêu đề hiển thị thông số kết quả tìm kiếm cho đồ thị trực quan hóa.
    
    Args:
        result: Kết quả tìm kiếm SearchResult.
        
    Returns:
        Chuỗi tiêu đề mô tả thuật toán, độ dài và số lượng node đã mở rộng.
    """
    if not result.found:
        return f"{result.algorithm}: Không có đường đi, duyệt={result.explored_nodes}"
    return (
        f"{result.algorithm}: Độ dài={result.path_length}, "
        f"duyệt={result.explored_nodes}, {result.elapsed_ms:.2f} ms"
    )

