"""Cấu hình dùng chung cho toàn bộ demo ANN + A* trên mê cung."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


# Thư mục gốc của project. Các đường dẫn bên dưới đều tính từ vị trí này.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Ba thư mục chính được pipeline sử dụng khi chạy lại từ đầu.
DATA_DIR = PROJECT_ROOT / "data"       # Lưu file dataset CSV.
MODELS_DIR = PROJECT_ROOT / "models"   # Lưu model Keras sau khi train.
OUTPUTS_DIR = PROJECT_ROOT / "outputs" # Lưu biểu đồ, metrics, ảnh demo.


@dataclass(frozen=True)
class MazeConfig:
    """Thông số mặc định khi sinh một mê cung ngẫu nhiên."""

    width: int = 20
    height: int = 20
    wall_probability: float = 0.25
    min_goal_distance: int = 20
    seed: int = 42


# 10 cột này chính là input của mạng ANN.
# Mỗi dòng dataset = một trạng thái trong mê cung được chuyển thành 10 số.
FEATURE_COLUMNS = [
    "x",           # Tọa độ cột hiện tại, đã chuẩn hóa về 0..1.
    "y",           # Tọa độ dòng hiện tại, đã chuẩn hóa về 0..1.
    "goal_x",      # Tọa độ cột của đích, đã chuẩn hóa.
    "goal_y",      # Tọa độ dòng của đích, đã chuẩn hóa.
    "dx",          # Khoảng cách ngang tới đích, đã chuẩn hóa.
    "dy",          # Khoảng cách dọc tới đích, đã chuẩn hóa.
    "up_wall",     # Ô phía trên có phải tường/ngoài biên không.
    "down_wall",   # Ô phía dưới có phải tường/ngoài biên không.
    "left_wall",   # Ô bên trái có phải tường/ngoài biên không.
    "right_wall",  # Ô bên phải có phải tường/ngoài biên không.
]

# Nhãn mà ANN phải học dự đoán: khoảng cách ngắn nhất thật sự tới đích.
TARGET_COLUMN = "true_distance"
