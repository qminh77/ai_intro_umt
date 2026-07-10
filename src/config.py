"""Cấu hình chung cho toàn bộ dự án Maze ANN A*.

File này chứa cấu hình đường dẫn thư mục dự án, cấu hình mặc định khi khởi tạo mê cung,
và định nghĩa các đặc trưng (features) đầu vào cùng nhãn mục tiêu (target) cho mạng ANN.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# Đường dẫn thư mục gốc của dự án
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Đường dẫn đến các thư mục con phục vụ lưu trữ dữ liệu, mô hình và kết quả đầu ra
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"


@dataclass(frozen=True)
class MazeConfig:
    """Cấu hình các tham số mặc định khi sinh mê cung ngẫu nhiên.
    
    Attributes:
        width: Chiều rộng của mê cung (số cột).
        height: Chiều cao của mê cung (số dòng).
        wall_probability: Xác suất xuất hiện tường (vật cản) trên mỗi ô (0.0 đến 1.0).
        min_goal_distance: Khoảng cách ngắn nhất tối thiểu giữa điểm xuất phát và đích.
        seed: Hạt giống ngẫu nhiên để tái lập kết quả.
    """
    width: int = 20
    height: int = 20
    wall_probability: float = 0.25
    min_goal_distance: int = 20
    seed: int = 42


# Danh sách 10 cột đặc trưng dùng làm đầu vào (input features) cho mô hình ANN
FEATURE_COLUMNS = [
    "x",           # Tọa độ X hiện tại của tác tử
    "y",           # Tọa độ Y hiện tại của tác tử
    "goal_x",      # Tọa độ X của điểm đích (Goal)
    "goal_y",      # Tọa độ Y của điểm đích (Goal)
    "dx",          # Khoảng cách tuyệt đối theo trục X giữa tác tử và đích
    "dy",          # Khoảng cách tuyệt đối theo trục Y giữa tác tử và đích
    "up_wall",     # Trạng thái ô phía trên (1 nếu là tường hoặc ngoài biên, ngược lại 0)
    "down_wall",   # Trạng thái ô phía dưới (1 nếu là tường hoặc ngoài biên, ngược lại 0)
    "left_wall",   # Trạng thái ô phía trái (1 nếu là tường hoặc ngoài biên, ngược lại 0)
    "right_wall",  # Trạng thái ô phía phải (1 nếu là tường hoặc ngoài biên, ngược lại 0)
]

# Tên cột chứa nhãn đích - khoảng cách thực tế ngắn nhất tìm được bằng BFS
TARGET_COLUMN = "true_distance"


