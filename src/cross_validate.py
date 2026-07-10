"""Script thực hiện đánh giá chéo K-Fold cho mô hình heuristic ANN.

File này chứa hàm phân tích cú pháp và hàm main để chia tập dữ liệu thành K phần (folds),
lần lượt huấn luyện trên K-1 phần và kiểm tra trên phần còn lại để đánh giá tính ổn định 
và độ lỗi trung bình (MAE, MSE) của mô hình học máy.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR, OUTPUTS_DIR
from .features import split_features_target
from .models import build_model, import_tensorflow
from .train import mean_absolute_error, mean_squared_error


def parse_args() -> argparse.Namespace:
    """Phân tích các đối số dòng lệnh đầu vào khi thực thi script cross-validation.
    
    Returns:
        argparse.Namespace chứa giá trị các tham số đã phân tích cú pháp.
    """
    parser = argparse.ArgumentParser(description="Đánh giá chéo K-Fold (Cross-Validation) cho mạng ANN.")
    parser.add_argument("--dataset", type=Path, default=DATA_DIR / "maze_dataset.csv", help="Đường dẫn file dữ liệu CSV.")
    parser.add_argument("--folds", type=int, default=5, help="Số lượng phần chia (K-Folds). Tối thiểu là 2.")
    parser.add_argument("--epochs", type=int, default=30, help="Số epoch huấn luyện cho mỗi fold.")
    parser.add_argument("--batch-size", type=int, default=64, help="Kích thước lô dữ liệu (batch size).")
    parser.add_argument("--seed", type=int, default=42, help="Hạt giống số ngẫu nhiên.")
    parser.add_argument("--output", type=Path, default=OUTPUTS_DIR / "cross_validation.json", help="Đường dẫn file JSON lưu kết quả.")
    return parser.parse_args()


def main() -> None:
    """Hàm chạy chính để khởi tạo tiến trình K-Fold Cross-Validation và xuất báo cáo kết quả."""
    args = parse_args()
    if args.folds < 2:
        raise ValueError("Số lượng folds phải từ 2 trở lên.")
    tf = import_tensorflow()

    # Đọc dữ liệu và chia đều các dòng vào các fold
    df = pd.read_csv(args.dataset)
    shuffled = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    fold_indices = np.array_split(np.arange(len(shuffled)), args.folds)
    folds = [shuffled.iloc[index].reset_index(drop=True) for index in fold_indices]

    results = []
    # Lần lượt lấy từng fold làm tập validation, các fold còn lại làm tập train
    for fold_index in range(args.folds):
        validation_df = folds[fold_index]
        train_df = pd.concat(
            [fold for index, fold in enumerate(folds) if index != fold_index],
            ignore_index=True,
        )

        # Tách đặc trưng và nhãn
        x_train, y_train = split_features_target(train_df)
        x_val, y_val = split_features_target(validation_df)

        # Thiết lập hạt giống ngẫu nhiên thay đổi nhẹ theo từng fold để đảm bảo khách quan
        tf.keras.utils.set_random_seed(args.seed + fold_index)
        
        # Xây dựng mô hình tốt nhất (goodfit)
        model = build_model("goodfit", input_dim=x_train.shape[1])
        model.fit(
            x_train,
            y_train,
            validation_data=(x_val, y_val),
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=0,
        )
        
        # Dự đoán và tính toán các chỉ số độ lỗi trên tập validation của fold này
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

    # Tính toán kết quả trung bình của tất cả các fold
    summary = {
        "folds": args.folds,
        "mean_mae": float(np.mean([item["mae"] for item in results])),
        "mean_mse": float(np.mean([item["mse"] for item in results])),
        "results": results,
    }

    # Lưu kết quả tổng hợp ra file JSON
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Đã lưu kết quả Cross-Validation tại: {args.output}")


if __name__ == "__main__":
    main()

