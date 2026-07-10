"""Huan luyen ANN: underfit, overfit, goodfit va cross-validation."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR, FEATURE_COLUMNS, MODELS_DIR, OUTPUTS_DIR, TARGET_COLUMN
from .dataset import split_features_target


@dataclass(frozen=True)
class ExperimentConfig:
    epochs: int
    batch_size: int
    train_limit: int | None
    early_stopping: bool
    description: str


EXPERIMENTS: dict[str, ExperimentConfig] = {
    "underfit": ExperimentConfig(
        epochs=5,
        batch_size=64,
        train_limit=None,
        early_stopping=False,
        description="Mang qua nho va train it epoch -> underfitting.",
    ),
    "overfit": ExperimentConfig(
        epochs=150,
        batch_size=32,
        train_limit=3000,
        early_stopping=False,
        description="Mang qua lon, du lieu train it, train lau -> overfitting.",
    ),
    "goodfit": ExperimentConfig(
        epochs=100,
        batch_size=64,
        train_limit=None,
        early_stopping=True,
        description="Mang vua phai, co Dropout va EarlyStopping -> fit tot hon.",
    ),
}


def import_tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "Can cai TensorFlow: python -m pip install -r requirements.txt"
        ) from exc
    return tf


def build_model(experiment: str, input_dim: int):
    tf = import_tensorflow()

    if experiment == "underfit":
        layers = [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(4, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    elif experiment == "overfit":
        layers = [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    elif experiment == "goodfit":
        layers = [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    else:
        raise ValueError(f"Khong co thi nghiem: {experiment}")

    model = tf.keras.Sequential(layers, name=f"ann_{experiment}_heuristic")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def training_callbacks(experiment: str):
    if not EXPERIMENTS[experiment].early_stopping:
        return []
    tf = import_tensorflow()
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        )
    ]


def split_dataframe(
    df: pd.DataFrame,
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    train_end = int(len(shuffled) * train_ratio)
    val_end = train_end + int(len(shuffled) * val_ratio)
    return shuffled.iloc[:train_end], shuffled.iloc[train_end:val_end], shuffled.iloc[val_end:]


def run_experiment(
    experiment: str,
    df: pd.DataFrame,
    seed: int,
    models_dir: Path,
    outputs_dir: Path,
    epochs_override: int | None = None,
) -> dict[str, float | int | str]:
    config = EXPERIMENTS[experiment]
    tf = import_tensorflow()
    tf.keras.utils.set_random_seed(seed)

    train_df, val_df, test_df = split_dataframe(df, seed=seed)
    if config.train_limit is not None and len(train_df) > config.train_limit:
        train_df = train_df.sample(n=config.train_limit, random_state=seed)

    x_train, y_train = split_features_target(train_df)
    x_val, y_val = split_features_target(val_df)
    x_test, y_test = split_features_target(test_df)

    model = build_model(experiment, input_dim=x_train.shape[1])
    epochs = epochs_override or config.epochs

    print(f"\n=== {experiment} ===")
    print(config.description)
    model.summary()
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=config.batch_size,
        callbacks=training_callbacks(experiment),
        verbose=2,
    )

    evaluation = model.evaluate(x_test, y_test, verbose=0, return_dict=True)
    predictions = model.predict(x_test, verbose=0).reshape(-1)
    test_mae = mean_absolute_error(y_test, predictions)
    test_mse = mean_squared_error(y_test, predictions)

    model_path = models_dir / f"ann_heuristic_{experiment}.keras"
    history_path = outputs_dir / f"{experiment}_history.csv"
    plot_path = outputs_dir / f"{experiment}_loss.png"
    metrics_path = outputs_dir / f"{experiment}_metrics.json"

    model.save(model_path)
    history_to_dataframe(history).to_csv(history_path, index=False)
    plot_training_history(history, plot_path, title=f"{experiment.title()} loss")

    metrics = {
        "experiment": experiment,
        "train_rows": int(len(train_df)),
        "validation_rows": int(len(val_df)),
        "test_rows": int(len(test_df)),
        "epochs_ran": int(len(history.history["loss"])),
        "test_loss_mse": float(evaluation["loss"]),
        "test_mae_from_keras": float(evaluation["mae"]),
        "test_mae": float(test_mae),
        "test_mse": float(test_mse),
        "model_path": str(model_path),
        "history_path": str(history_path),
        "plot_path": str(plot_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Da luu model: {model_path}")
    print(f"Test MAE: {test_mae:.3f} buoc")
    return metrics


def run_cross_validation(
    df: pd.DataFrame,
    folds: int,
    epochs: int,
    batch_size: int,
    seed: int,
    output_path: Path,
) -> dict[str, object]:
    if folds < 2:
        raise ValueError("folds phai >= 2")

    tf = import_tensorflow()
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    fold_indices = np.array_split(np.arange(len(shuffled)), folds)
    fold_frames = [shuffled.iloc[index].reset_index(drop=True) for index in fold_indices]

    results = []
    for fold_index in range(folds):
        validation_df = fold_frames[fold_index]
        train_df = pd.concat(
            [fold for index, fold in enumerate(fold_frames) if index != fold_index],
            ignore_index=True,
        )
        x_train, y_train = split_features_target(train_df)
        x_val, y_val = split_features_target(validation_df)

        tf.keras.utils.set_random_seed(seed + fold_index)
        model = build_model("goodfit", input_dim=x_train.shape[1])
        model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=training_callbacks("goodfit"),
            verbose=0,
        )

        predictions = model.predict(x_val, verbose=0).reshape(-1)
        fold_result = {
            "fold": fold_index + 1,
            "validation_rows": int(len(validation_df)),
            "mae": mean_absolute_error(y_val, predictions),
            "mse": mean_squared_error(y_val, predictions),
        }
        results.append(fold_result)
        print(
            f"Fold {fold_index + 1}/{folds}: "
            f"MAE={fold_result['mae']:.3f}, MSE={fold_result['mse']:.3f}"
        )

    summary = {
        "folds": folds,
        "epochs": epochs,
        "mean_mae": float(np.mean([item["mae"] for item in results])),
        "mean_mse": float(np.mean([item["mse"] for item in results])),
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Da luu cross-validation: {output_path}")
    return summary


def history_to_dataframe(history) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "epoch": np.arange(1, len(history.history["loss"]) + 1),
            **{key: values for key, values in history.history.items()},
        }
    )


def plot_training_history(history, output_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    epochs = np.arange(1, len(history.history["loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history.history["loss"], label="Train loss")
    axes[0].plot(epochs, history.history["val_loss"], label="Validation loss")
    axes[0].set_title(title)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history.history["mae"], label="Train MAE")
    axes[1].plot(epochs, history.history["val_mae"], label="Validation MAE")
    axes[1].set_title("MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Sai so trung binh (buoc)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.square(y_true - y_pred)))


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Khong tim thay dataset: {path}. Hay chay: python -m src.dataset")
    df = pd.read_csv(path)
    missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset thieu cot: {sorted(missing)}")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ANN heuristic va chay cross-validation.")
    parser.add_argument("--dataset", type=Path, default=DATA_DIR / "maze_dataset.csv")
    parser.add_argument("--experiment", choices=["underfit", "overfit", "goodfit", "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None, help="Ghi de so epoch khi can test nhanh.")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--cross-val", action="store_true", help="Chay K-fold cross-validation cho goodfit.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cv-output", type=Path, default=OUTPUTS_DIR / "cross_validation.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_dataset(args.dataset)

    if args.cross_val:
        run_cross_validation(
            df=df,
            folds=args.folds,
            epochs=args.epochs or 10,
            batch_size=args.batch_size,
            seed=args.seed,
            output_path=args.cv_output,
        )
        return

    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.outputs_dir.mkdir(parents=True, exist_ok=True)
    experiments = list(EXPERIMENTS) if args.experiment == "all" else [args.experiment]
    metrics = {
        experiment: run_experiment(
            experiment=experiment,
            df=df,
            seed=args.seed,
            models_dir=args.models_dir,
            outputs_dir=args.outputs_dir,
            epochs_override=args.epochs,
        )
        for experiment in experiments
    }

    summary_path = args.outputs_dir / "training_summary.json"
    summary_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Da luu tom tat train: {summary_path}")


if __name__ == "__main__":
    main()
