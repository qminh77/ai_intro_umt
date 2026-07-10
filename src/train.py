"""Huấn luyện ANN: underfit, overfit, goodfit và cross-validation.

Đây là file quan trọng nhất cho phần thực nghiệm. Nếu muốn tự thay số layer,
số node hoặc số epoch để tạo underfitting/overfitting, chỉnh trong file này.
"""

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
    """Cấu hình huấn luyện cho một thí nghiệm."""

    epochs: int
    batch_size: int
    train_limit: int | None
    early_stopping: bool
    description: str


# Chỉnh số epoch ở đây nếu muốn thay đổi thời gian học mặc định.
# train_limit dùng để cố tình giảm dữ liệu train, giúp tạo overfitting dễ hơn.
EXPERIMENTS: dict[str, ExperimentConfig] = {
    "underfit": ExperimentConfig(
        epochs=5,  # Ít epoch để mô hình chưa học đủ.
        batch_size=64,
        train_limit=None,
        early_stopping=False,
        description="Mạng quá nhỏ và train ít epoch -> underfitting.",
    ),
    "overfit": ExperimentConfig(
        epochs=150,  # Nhiều epoch để mô hình có cơ hội học vẹt.
        batch_size=32,
        train_limit=3000,  # Giới hạn dữ liệu train để overfit xuất hiện rõ hơn.
        early_stopping=False,
        description="Mạng quá lớn, dữ liệu train ít, train lâu -> overfitting.",
    ),
    "goodfit": ExperimentConfig(
        epochs=100,  # Đây là số epoch tối đa; EarlyStopping có thể dừng sớm hơn.
        batch_size=64,
        train_limit=None,
        early_stopping=True,
        description="Mạng vừa phải, có Dropout và EarlyStopping -> fit tốt hơn.",
    ),
}


def import_tensorflow():
    # Import TensorFlow ở trong hàm để các lệnh không cần train vẫn load nhanh hơn.
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "Cần cài TensorFlow: python -m pip install -r requirements.txt"
        ) from exc
    return tf


def build_model(experiment: str, input_dim: int):
    """Tạo kiến trúc ANN theo tên thí nghiệm.

    Đây là nơi chỉnh số layer và số node. Mỗi dòng `Dense(...)` là một layer
    fully-connected. Số trong `Dense(64)` chính là số node/neuron của layer đó.
    """

    tf = import_tensorflow()

    if experiment == "underfit":
        # UNDERFIT: mạng cố tình rất nhỏ nên khó học hết quy luật của dữ liệu.
        layers = [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(4, activation="relu"),  # Chỉnh 4 -> 2 để underfit nặng hơn.
            tf.keras.layers.Dense(1),                     # Output: dự đoán true_distance.
        ]
    elif experiment == "overfit":
        # OVERFIT: mạng rất lớn, nhiều tham số, dễ học vẹt khi train trên ít dữ liệu.
        layers = [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(512, activation="relu"),
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    elif experiment == "goodfit":
        # GOODFIT: mạng vừa phải. Dropout tắt ngẫu nhiên một phần node khi train
        # để giảm học vẹt, EarlyStopping được bật ở phần callback bên dưới.
        layers = [
            tf.keras.layers.Input(shape=(input_dim,)),
            tf.keras.layers.Dense(64, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(32, activation="relu"),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(1),
        ]
    else:
        raise ValueError(f"Không có thí nghiệm: {experiment}")

    model = tf.keras.Sequential(layers, name=f"ann_{experiment}_heuristic")

    # Vì output là khoảng cách dạng số, đây là bài toán hồi quy, không phải phân loại.
    # Do đó loss dùng MSE và metric dễ hiểu hơn là MAE: sai trung bình bao nhiêu bước.
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def training_callbacks(experiment: str):
    # Chỉ goodfit dùng EarlyStopping. Underfit/overfit cố tình không dùng để dễ minh họa hiện tượng.
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
    # Chia dữ liệu theo tỉ lệ 70/15/15: train/validation/test.
    # Validation dùng trong lúc train, test chỉ dùng để đánh giá cuối cùng.
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

    # Với overfit, ta cố tình train trên ít dòng hơn để mô hình dễ học thuộc tập train.
    if config.train_limit is not None and len(train_df) > config.train_limit:
        train_df = train_df.sample(n=config.train_limit, random_state=seed)

    # Tách dữ liệu thành input X và nhãn y cho Keras.
    x_train, y_train = split_features_target(train_df)
    x_val, y_val = split_features_target(val_df)
    x_test, y_test = split_features_target(test_df)

    model = build_model(experiment, input_dim=x_train.shape[1])
    epochs = epochs_override or config.epochs

    print(f"\n=== {experiment} ===")
    print(config.description)
    model.summary()

    # model.fit là vòng lặp học chính: forward -> tính loss -> backpropagation -> cập nhật trọng số.
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=epochs,
        batch_size=config.batch_size,
        callbacks=training_callbacks(experiment),
        verbose=2,
    )

    # Test set chỉ dùng sau khi train xong để đánh giá khách quan.
    evaluation = model.evaluate(x_test, y_test, verbose=0, return_dict=True)
    predictions = model.predict(x_test, verbose=0).reshape(-1)
    test_mae = mean_absolute_error(y_test, predictions)
    test_mse = mean_squared_error(y_test, predictions)

    model_path = models_dir / f"ann_heuristic_{experiment}.keras"
    history_path = outputs_dir / f"{experiment}_history.csv"
    plot_path = outputs_dir / f"{experiment}_loss.png"
    metrics_path = outputs_dir / f"{experiment}_metrics.json"

    # Lưu lại model, lịch sử học và biểu đồ để đưa vào báo cáo/slide.
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
    print(f"Đã lưu model: {model_path}")
    print(f"Test MAE: {test_mae:.3f} bước")
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
        raise ValueError("folds phải >= 2")

    tf = import_tensorflow()

    # Cross-validation: chia dataset thành K phần, lần lượt lấy 1 phần làm validation.
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    fold_indices = np.array_split(np.arange(len(shuffled)), folds)
    fold_frames = [shuffled.iloc[index].reset_index(drop=True) for index in fold_indices]

    results = []
    for fold_index in range(folds):
        # Fold hiện tại là validation, các fold còn lại ghép thành train.
        validation_df = fold_frames[fold_index]
        train_df = pd.concat(
            [fold for index, fold in enumerate(fold_frames) if index != fold_index],
            ignore_index=True,
        )
        x_train, y_train = split_features_target(train_df)
        x_val, y_val = split_features_target(validation_df)

        tf.keras.utils.set_random_seed(seed + fold_index)

        # Cross-validation chỉ kiểm tra độ ổn định của model goodfit.
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
    print(f"Đã lưu cross-validation: {output_path}")
    return summary


def history_to_dataframe(history) -> pd.DataFrame:
    # Keras trả về history dạng dict. Chuyển sang CSV để dễ mở bằng Excel/Sheets.
    return pd.DataFrame(
        {
            "epoch": np.arange(1, len(history.history["loss"]) + 1),
            **{key: values for key, values in history.history.items()},
        }
    )


def plot_training_history(history, output_path: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    # Biểu đồ quan trọng nhất để nhìn underfit/overfit là train loss và validation loss.
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
    axes[1].set_ylabel("Sai số trung bình (bước)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # MAE: sai số tuyệt đối trung bình, đơn vị là số bước trong mê cung.
    return float(np.mean(np.abs(y_true - y_pred)))


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # MSE phạt lỗi lớn mạnh hơn MAE vì có bình phương sai số.
    return float(np.mean(np.square(y_true - y_pred)))


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy dataset: {path}. Hãy chạy: python -m src.dataset")
    df = pd.read_csv(path)
    missing = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset thiếu cột: {sorted(missing)}")
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ANN heuristic và chạy cross-validation.")
    parser.add_argument("--dataset", type=Path, default=DATA_DIR / "maze_dataset.csv")
    parser.add_argument("--experiment", choices=["underfit", "overfit", "goodfit", "all"], default="all")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=None, help="Ghi đè số epoch khi cần test nhanh.")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_DIR)
    parser.add_argument("--cross-val", action="store_true", help="Chạy K-fold cross-validation cho goodfit.")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cv-output", type=Path, default=OUTPUTS_DIR / "cross_validation.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_dataset(args.dataset)

    # Nếu có --cross-val thì chỉ chạy cross-validation, không train/lưu 3 model chính.
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

    # --experiment all sẽ chạy cả underfit, overfit và goodfit.
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
    print(f"Đã lưu tóm tắt train: {summary_path}")


if __name__ == "__main__":
    main()
