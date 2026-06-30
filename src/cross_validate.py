from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR, OUTPUTS_DIR
from .features import split_features_target
from .models import build_model
from .train import mean_absolute_error, mean_squared_error


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Simple k-fold cross-validation for ANN.")
    parser.add_argument("--dataset", type=Path, default=DATA_DIR / "maze_dataset.csv")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=OUTPUTS_DIR / "cross_validation.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.folds < 2:
        raise ValueError("--folds must be at least 2.")

    df = pd.read_csv(args.dataset)
    shuffled = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    fold_indices = np.array_split(np.arange(len(shuffled)), args.folds)
    folds = [shuffled.iloc[index].reset_index(drop=True) for index in fold_indices]

    results = []
    for fold_index in range(args.folds):
        validation_df = folds[fold_index]
        train_df = pd.concat(
            [fold for index, fold in enumerate(folds) if index != fold_index],
            ignore_index=True,
        )

        x_train, y_train = split_features_target(train_df)
        x_val, y_val = split_features_target(validation_df)

        model = build_model("goodfit", input_dim=x_train.shape[1])
        model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=0,
        )
        predictions = model.predict(x_val, verbose=0).reshape(-1)
        results.append(
            {
                "fold": fold_index + 1,
                "validation_rows": int(len(validation_df)),
                "mae": mean_absolute_error(y_val, predictions),
                "mse": mean_squared_error(y_val, predictions),
            }
        )
        print(
            f"Fold {fold_index + 1}/{args.folds}: "
            f"MAE={results[-1]['mae']:.3f}, MSE={results[-1]['mse']:.3f}"
        )

    summary = {
        "folds": args.folds,
        "mean_mae": float(np.mean([item["mae"] for item in results])),
        "mean_mse": float(np.mean([item["mse"] for item in results])),
        "results": results,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved cross-validation summary: {args.output}")


if __name__ == "__main__":
    main()
