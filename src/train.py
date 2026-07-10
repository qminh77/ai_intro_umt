"""Script huấn luyện và đánh giá các mô hình Heuristic ANN trên tập dữ liệu mê cung.

File này chịu trách nhiệm load dữ liệu CSV, chia dữ liệu thành các tập Train/Val/Test, 
xây dựng mô hình Keras tương ứng với cấu hình thí nghiệm, tiến hành train mô hình, 
chuyển đổi và lưu trữ mô hình dưới dạng .keras và .tflite, đồng thời vẽ biểu đồ lịch sử huấn luyện.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR, FEATURE_COLUMNS, MODELS_DIR, OUTPUTS_DIR, TARGET_COLUMN
from .features import split_features_target
from .models import EXPERIMENTS, build_model, import_tensorflow, training_callbacks


def parse_args() -> argparse.Namespace:
    """Phân tích các đối số dòng lệnh đầu vào khi thực thi script huấn luyện.
    
    Returns:
        argparse.Namespace chứa giá trị các tham số đã phân tích cú pháp.
    """
    parser = argparse.ArgumentParser(description="Huấn luyện các mô hình heuristic ANN theo kịch bản thí nghiệm.")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATA_DIR / "maze_dataset.csv",
        help="Đường dẫn đến file CSV chứa tập dữ liệu mê cung đã sinh.",
    )
    parser.add_argument(
        "--experiment",
        choices=["underfit", "overfit", "goodfit", "all"],
        default="all",
        help="Chọn thí nghiệm huấn luyện cụ thể, hoặc 'all' để huấn luyện cả 3.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Hạt giống số ngẫu nhiên.")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR, help="Thư mục lưu trữ file mô hình đầu ra.")
    parser.add_argument("--outputs-dir", type=Path, default=OUTPUTS_DIR, help="Thư mục lưu trữ biểu đồ và tóm tắt kết quả.")
    return parser.parse_args()


def main() -> None:
    """Hàm điều phối chính cho quá trình chạy toàn bộ pipeline huấn luyện mô hình."""
    args = parse_args()
    args.models_dir.mkdir(parents=True, exist_ok=True)
    args.outputs_dir.mkdir(parents=True, exist_ok=True)

    # Đảm bảo tập dữ liệu CSV đầu vào đã được sinh trước đó
    if not args.dataset.exists():
        raise FileNotFoundError(
            f"Không tìm thấy tập dữ liệu: {args.dataset}. Hãy chạy script sinh dữ liệu trước: `python -m src.generate_dataset`."
        )

    # Đọc dữ liệu và xác thực cấu trúc các cột đặc trưng/nhãn
    df = pd.read_csv(args.dataset)
    missing_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN]) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Tập dữ liệu bị thiếu các cột bắt buộc: {sorted(missing_columns)}")

    # Lập danh sách các thí nghiệm cần chạy dựa trên lựa chọn đầu vào
    experiments = list(EXPERIMENTS) if args.experiment == "all" else [args.experiment]
    metrics = {}

    for experiment in experiments:
        print(f"\n=== Bắt đầu huấn luyện thí nghiệm: {experiment} ===")
        metrics[experiment] = run_experiment(
            experiment=experiment,
            df=df,
            seed=args.seed,
            models_dir=args.models_dir,
            outputs_dir=args.outputs_dir,
        )

    # Lưu tóm tắt kết quả kiểm định của tất cả thí nghiệm ra file JSON
    summary_path = args.outputs_dir / "training_summary.json"
    summary_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"\nĐã lưu tóm tắt kết quả huấn luyện tại: {summary_path}")


def run_experiment(
    experiment: str,
    df: pd.DataFrame,
    seed: int,
    models_dir: Path,
    outputs_dir: Path,
) -> dict[str, float | int | str]:
    """Thực thi toàn bộ luồng huấn luyện cho một thí nghiệm cụ thể.
    
    Các bước bao gồm: Chia dữ liệu -> Giới hạn dữ liệu nếu cấu hình yêu cầu -> Tách đặc trưng & Nhãn ->
    Xây dựng mô hình -> Huấn luyện với Callbacks -> Đánh giá trên tập kiểm thử Test -> 
    Lưu mô hình Keras & TFLite -> Lưu lịch sử & Biểu đồ.
    
    Args:
        experiment: Tên thí nghiệm cần chạy.
        df: DataFrame chứa toàn bộ dữ liệu.
        seed: Hạt giống số ngẫu nhiên.
        models_dir: Thư mục lưu mô hình.
        outputs_dir: Thư mục lưu biểu đồ/kết quả.
        
    Returns:
        Dictionary chứa các chỉ số đánh giá hiệu năng mô hình trên tập Test và đường dẫn các file kết quả.
    """
    config = EXPERIMENTS[experiment]
    tf = import_tensorflow()
    tf.keras.utils.set_random_seed(seed)
    
    # Chia dữ liệu theo tỷ lệ chuẩn 70% Train, 15% Validation, 15% Test
    train_df, val_df, test_df = split_dataframe(df, seed=seed)

    # Nếu cấu hình thí nghiệm yêu cầu giới hạn số mẫu train (ví dụ kịch bản Overfit)
    if config.train_limit is not None and len(train_df) > config.train_limit:
        train_df = train_df.sample(n=config.train_limit, random_state=seed)

    # Tách ma trận đặc trưng X và nhãn Y cho từng tập
    x_train, y_train = split_features_target(train_df)
    x_val, y_val = split_features_target(val_df)
    x_test, y_test = split_features_target(test_df)

    # Khởi tạo mô hình Keras Sequential
    model = build_model(experiment, input_dim=x_train.shape[1])
    print(config.description)
    model.summary()

    # Huấn luyện mô hình
    history = model.fit(
        x_train,
        y_train,
        validation_data=(x_val, y_val),
        epochs=config.epochs,
        batch_size=config.batch_size,
        callbacks=training_callbacks(experiment),
        verbose=2,
    )

    # Đánh giá hiệu năng mô hình trên tập test độc lập
    evaluation = model.evaluate(x_test, y_test, verbose=0, return_dict=True)
    predictions = model.predict(x_test, verbose=0).reshape(-1)
    test_mae = mean_absolute_error(y_test, predictions)
    test_mse = mean_squared_error(y_test, predictions)

    # Xác định đường dẫn lưu trữ các file đầu ra của thí nghiệm
    model_path = models_dir / f"ann_heuristic_{experiment}.keras"
    tflite_path = models_dir / f"ann_heuristic_{experiment}.tflite"
    history_path = outputs_dir / f"{experiment}_history.csv"
    plot_path = outputs_dir / f"{experiment}_loss.png"
    metrics_path = outputs_dir / f"{experiment}_metrics.json"

    # 1. Lưu mô hình định dạng Keras nguyên bản
    model.save(model_path)

    # 2. Chuyển đổi sang định dạng TensorFlow Lite (TFLite) để suy luận nhanh
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    tflite_path.write_bytes(tflite_model)

    # 3. Lưu lịch sử huấn luyện thành file CSV và vẽ biểu đồ loss/metrics
    history_to_dataframe(history).to_csv(history_path, index=False)
    plot_training_history(history, plot_path, title=f"Thí nghiệm {experiment.title()}: Hàm mất mát (Loss)")

    # Thu thập toàn bộ chỉ số để đóng gói trả về và lưu ra JSON
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
        "tflite_path": str(tflite_path),
        "history_path": str(history_path),
        "plot_path": str(plot_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Đã lưu mô hình Keras: {model_path}")
    print(f"Đã lưu mô hình TFLite: {tflite_path}")
    print(f"Đã lưu biểu đồ: {plot_path}")
    print(f"Kết quả Test MAE: {test_mae:.3f} steps")
    return metrics


def split_dataframe(
    df: pd.DataFrame,
    seed: int,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Phân tách ngẫu nhiên một DataFrame thành 3 tập dữ liệu: Train, Validation và Test.
    
    Tỷ lệ phân chia mặc định: 70% Train, 15% Validation, 15% Test.
    
    Args:
        df: Pandas DataFrame gốc.
        seed: Hạt giống số ngẫu nhiên dùng để trộn.
        train_ratio: Tỷ lệ tập Train.
        val_ratio: Tỷ lệ tập Validation.
        
    Returns:
        Tuple (train_df, val_df, test_df) chứa 3 phân mục dữ liệu.
    """
    shuffled = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    train_end = int(len(shuffled) * train_ratio)
    val_end = train_end + int(len(shuffled) * val_ratio)
    return shuffled.iloc[:train_end], shuffled.iloc[train_end:val_end], shuffled.iloc[val_end:]


def history_to_dataframe(history) -> pd.DataFrame:
    """Chuyển đổi đối tượng lịch sử huấn luyện của Keras thành một Pandas DataFrame.
    
    Args:
        history: Đối tượng trả về từ hàm model.fit().
        
    Returns:
        Pandas DataFrame chứa thông tin loss và metrics qua từng epoch.
    """
    rows = []
    for epoch_index in range(len(history.history["loss"])):
        row = {"epoch": epoch_index + 1}
        for key, values in history.history.items():
            row[key] = float(values[epoch_index])
        rows.append(row)
    return pd.DataFrame(rows)


def plot_training_history(history, output_path: Path, title: str) -> None:
    """Vẽ và lưu trữ biểu đồ lịch sử quá trình huấn luyện mô hình.
    
    Vẽ song song 2 biểu đồ: MSE loss (hàm mất mát) và MAE metric (số bước sai số) 
    cho cả tập huấn luyện (Train) và tập kiểm định (Validation) qua từng Epoch.
    
    Args:
        history: Đối tượng lịch sử huấn luyện Keras.
        output_path: Đường dẫn lưu ảnh biểu đồ PNG.
        title: Tiêu đề chính hiển thị trên biểu đồ loss.
        
    Raises:
        RuntimeError: Nếu chưa cài đặt thư viện matplotlib.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "Yêu cầu cài đặt matplotlib để vẽ biểu đồ lịch sử huấn luyện. Hãy cài đặt các thư viện cần thiết."
        ) from exc

    epochs = np.arange(1, len(history.history["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    # Biểu đồ trái: MSE Loss
    axes[0].plot(epochs, history.history["loss"], label="Train loss")
    axes[0].plot(epochs, history.history["val_loss"], label="Validation loss")
    axes[0].set_title(title)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Biểu đồ phải: MAE Metric
    axes[1].plot(epochs, history.history["mae"], label="Train MAE")
    axes[1].plot(epochs, history.history["val_mae"], label="Validation MAE")
    axes[1].set_title("Chỉ số MAE")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Sai số trung bình (số bước)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Tính toán sai số tuyệt đối trung bình (MAE) giữa nhãn thực tế và dự đoán.
    
    MAE = (1 / n) * sum(|y_true - y_pred|)
    
    Args:
        y_true: Mảng NumPy chứa nhãn thực tế.
        y_pred: Mảng NumPy chứa nhãn dự đoán.
        
    Returns:
        Giá trị MAE (số thực).
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Tính toán sai số bình phương trung bình (MSE) giữa nhãn thực tế và dự đoán.
    
    MSE = (1 / n) * sum((y_true - y_pred)^2)
    
    Args:
        y_true: Mảng NumPy chứa nhãn thực tế.
        y_pred: Mảng NumPy chứa nhãn dự đoán.
        
    Returns:
        Giá trị MSE (số thực).
    """
    return float(np.mean(np.square(y_true - y_pred)))


if __name__ == "__main__":
    main()
