"""Định nghĩa cấu hình thí nghiệm và cấu trúc các mô hình học máy ANN.

File này chứa dataclass lưu trữ tham số huấn luyện của từng thí nghiệm (underfit, overfit, goodfit),
hàm xây dựng kiến trúc mạng Keras tương ứng cho từng thí nghiệm, các callback huấn luyện,
và phương thức import an toàn cho thư viện TensorFlow.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ExperimentConfig:
    """Dataclass chứa cấu hình siêu tham số (hyperparameters) cho từng kịch bản huấn luyện mô hình.
    
    Attributes:
        name: Tên của thí nghiệm (underfit, overfit, goodfit).
        epochs: Số lượng chu kỳ huấn luyện (epochs) tối đa.
        batch_size: Kích thước lô dữ liệu (batch size) đưa vào mô hình mỗi lần cập nhật trọng số.
        train_limit: Giới hạn số dòng dữ liệu dùng để huấn luyện (để giả lập thiếu dữ liệu, ví dụ overfit).
        early_stopping: Có kích hoạt cơ chế dừng sớm (Early Stopping) khi validation loss không cải thiện hay không.
        description: Mô tả chi tiết mục đích và ý nghĩa của thí nghiệm.
    """
    name: str
    epochs: int
    batch_size: int
    train_limit: int | None
    early_stopping: bool
    description: str


# Định nghĩa 3 kịch bản thí nghiệm để minh họa các vấn đề cơ bản trong Machine Learning
EXPERIMENTS: dict[str, ExperimentConfig] = {
    "underfit": ExperimentConfig(
        name="underfit",
        epochs=5,
        batch_size=64,
        train_limit=None,
        early_stopping=False,
        description="Mạng nơ-ron cực kỳ đơn giản (4 node ẩn) kết hợp số lượng Epochs quá nhỏ để minh họa hiện tượng Underfitting.",
    ),
    "overfit": ExperimentConfig(
        name="overfit",
        epochs=150,
        batch_size=32,
        train_limit=3000,  # Giới hạn tập train rất nhỏ (3000 mẫu) để mô hình dễ "học vẹt"
        early_stopping=False,
        description="Mạng nơ-ron rất sâu/rộng huấn luyện quá nhiều epoch trên một tập dữ liệu nhỏ để minh họa hiện tượng Overfitting.",
    ),
    "goodfit": ExperimentConfig(
        name="goodfit",
        epochs=100,
        batch_size=64,
        train_limit=None,
        early_stopping=True,  # Kích hoạt Early Stopping để tự động dừng khi mô hình đạt điểm tối ưu
        description="Mạng nơ-ron có kích thước vừa phải, tích hợp lớp Dropout và cơ chế dừng sớm Early Stopping để đạt kết quả tốt nhất.",
    ),
}


def import_tensorflow():
    """Import thư viện TensorFlow một cách an toàn và ném ra lỗi dễ đọc nếu chưa cài đặt.
    
    Returns:
        Module tensorflow.
        
    Raises:
        RuntimeError: Nếu không thể import thư viện tensorflow.
    """
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "Yêu cầu cài đặt TensorFlow để thực hiện huấn luyện mô hình. Hãy cài đặt thông qua file requirements.txt: "
            "`python -m pip install -r requirements.txt`."
        ) from exc
    return tf


def build_model(experiment: str, input_dim: int):
    """Xây dựng và biên dịch mô hình Keras Sequential dựa trên tên thí nghiệm được chỉ định.
    
    Args:
        experiment: Tên thí nghiệm ("underfit", "overfit", "goodfit").
        input_dim: Số lượng đặc trưng đầu vào (kích thước chiều của X, thông thường là 10).
        
    Returns:
        Mô hình Keras Sequential đã được compile với Optimizer Adam và Loss MSE.
        
    Raises:
        ValueError: Nếu tên thí nghiệm không hợp lệ.
    """
    tf = import_tensorflow()

    if experiment == "underfit":
        # Kiến trúc cực nông để gây hiện tượng Underfit
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(input_dim,)),
                tf.keras.layers.Dense(4, activation="relu"),
                tf.keras.layers.Dense(1),
            ],
            name="ann_underfit_heuristic",
        )
    elif experiment == "overfit":
        # Kiến trúc cực sâu, nhiều neuron ẩn để gây Overfit khi dữ liệu ít
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(input_dim,)),
                tf.keras.layers.Dense(512, activation="relu"),
                tf.keras.layers.Dense(512, activation="relu"),
                tf.keras.layers.Dense(256, activation="relu"),
                tf.keras.layers.Dense(128, activation="relu"),
                tf.keras.layers.Dense(1),
            ],
            name="ann_overfit_heuristic",
        )
    elif experiment == "goodfit":
        # Kiến trúc cân đối, có lớp Dropout điều hòa tránh Overfit
        model = tf.keras.Sequential(
            [
                tf.keras.layers.Input(shape=(input_dim,)),
                tf.keras.layers.Dense(64, activation="relu"),
                tf.keras.layers.Dropout(0.2),
                tf.keras.layers.Dense(32, activation="relu"),
                tf.keras.layers.Dense(16, activation="relu"),
                tf.keras.layers.Dense(1),
            ],
            name="ann_goodfit_heuristic",
        )
    else:
        raise ValueError(f"Không tìm thấy cấu hình cho thí nghiệm: {experiment}")

    # Biên dịch mô hình sử dụng hàm tối ưu Adam, hàm mất mát MSE và đánh giá bằng MAE
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="mse",
        metrics=[tf.keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def training_callbacks(experiment: str):
    """Lấy danh sách các Callback Keras dựa trên thí nghiệm hiện tại.
    
    Chỉ kích hoạt Early Stopping cho thí nghiệm "goodfit".
    
    Args:
        experiment: Tên thí nghiệm.
        
    Returns:
        Danh sách các Keras callbacks (có thể rỗng).
    """
    tf = import_tensorflow()
    config = EXPERIMENTS[experiment]

    if not config.early_stopping:
        return []

    # Dừng sớm nếu validation loss không giảm sau 10 epoch liên tục, đồng thời khôi phục trọng số tốt nhất
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            restore_best_weights=True,
        )
    ]


