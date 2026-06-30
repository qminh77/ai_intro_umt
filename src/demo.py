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
    parser = argparse.ArgumentParser(description="Compare BFS, A* Manhattan, and A* ANN.")
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--wall-prob", type=float, default=0.25)
    parser.add_argument("--min-goal-distance", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--model",
        type=Path,
        default=MODELS_DIR / "ann_heuristic_goodfit.keras",
        help="Trained Keras model. If absent, A* ANN is skipped.",
    )
    parser.add_argument("--output", type=Path, default=OUTPUTS_DIR / "demo_path.png")
    parser.add_argument("--metrics", type=Path, default=OUTPUTS_DIR / "demo_metrics.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    maze, goal, distances = generate_training_maze(
        height=args.height,
        width=args.width,
        wall_probability=args.wall_prob,
        min_goal_distance=args.min_goal_distance,
        rng=rng,
    )
    start = choose_demo_start(distances, min_distance=args.min_goal_distance, rng=rng)

    results = [
        bfs_path(maze, start, goal),
        astar(maze, start, goal, manhattan, algorithm_name="A* + Manhattan"),
    ]

    if args.model.exists():
        ann_heuristic = AnnHeuristic(maze=maze, model_path=args.model)
        results.append(
            astar(maze, start, goal, ann_heuristic, algorithm_name="A* + ANN")
        )
    else:
        print(f"Model not found, skipping A* ANN: {args.model}")

    visual_output = save_search_figure(maze, start, goal, results, args.output)
    save_metrics(results, args.metrics, start=start, goal=goal)

    print(f"Start: {start}, Goal: {goal}")
    for result in results:
        status = "found" if result.found else "not found"
        print(
            f"{result.algorithm}: {status}, length={result.path_length}, "
            f"explored={result.explored_nodes}, time={result.elapsed_ms:.2f} ms"
        )
    print(f"Saved visual output: {visual_output}")
    print(f"Saved metrics: {args.metrics}")


def choose_demo_start(
    distances: np.ndarray,
    min_distance: int,
    rng: np.random.Generator,
) -> Position:
    candidates = np.argwhere(np.isfinite(distances) & (distances >= min_distance))
    if len(candidates) == 0:
        candidates = np.argwhere(np.isfinite(distances))
    selected = candidates[int(rng.integers(0, len(candidates)))]
    return int(selected[0]), int(selected[1])


def save_metrics(
    results,
    output_path: Path,
    start: Position,
    goal: Position,
) -> None:
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
