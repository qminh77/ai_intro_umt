from __future__ import annotations

import argparse
from pathlib import Path

from .config import DATA_DIR
from .dataset import generate_labeled_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate BFS-labeled maze dataset.")
    parser.add_argument("--mazes", type=int, default=500, help="Number of mazes to generate.")
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

    print(f"Saved dataset: {args.output}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {', '.join(df.columns)}")


if __name__ == "__main__":
    main()

