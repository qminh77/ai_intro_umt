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
    """Save visual comparison. Falls back to ASCII if matplotlib is unavailable."""
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
        axis.imshow(maze == WALL, cmap=cmap, origin="upper")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(_title_for_result(result), fontsize=11, pad=10)

        if result.path:
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
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Matplotlib is not installed, so this ASCII fallback was generated.",
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
    if not result.found:
        return f"{result.algorithm}: no path, explored={result.explored_nodes}"
    return (
        f"{result.algorithm}: length={result.path_length}, "
        f"explored={result.explored_nodes}, {result.elapsed_ms:.2f} ms"
    )
