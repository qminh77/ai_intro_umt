"""Lớp wrapper tích hợp mô hình TensorFlow Lite làm hàm Heuristic cho A*.

File này định nghĩa lớp AnnHeuristic, chịu trách nhiệm tải mô hình .tflite, phân tích 
và chuyển đổi trạng thái mê cung hiện tại thành các đặc trưng đầu vào tương thích,
thực thi suy luận thông qua interpreter của TFLite và cache kết quả để tối ưu tốc độ.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .features import state_to_features
from .maze import Position
from .models import import_tensorflow


class AnnHeuristic:
    """Bộ bọc (wrapper) mô hình TensorFlow Lite để truyền vào làm hàm heuristic h(n) trong A*."""

    def __init__(self, maze: np.ndarray, model_path: Path) -> None:
        """Khởi tạo bộ Heuristic ANN bằng cách load mô hình TFLite và chuẩn bị bộ thông dịch (interpreter).
        
        Args:
            maze: Ma trận mê cung 2D hiện tại.
            model_path: Đường dẫn trỏ tới file mô hình .tflite đã được huấn luyện.
        """
        tf = import_tensorflow()
        self.maze = maze
        
        # Load mô hình TFLite và cấp phát bộ nhớ cho các tensor đầu vào/đầu ra
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        
        # Cache kết quả dự đoán tránh tính toán lại nhiều lần cho cùng một cặp (vị trí, đích)
        self.cache: dict[tuple[Position, Position], float] = {}

    def __call__(self, position: Position, goal: Position) -> float:
        """Ước lượng khoảng cách từ vị trí hiện tại tới Goal bằng mô hình ANN.
        
        Phương thức này được gọi bởi thuật toán A* tại mỗi bước khám phá nút lân cận.
        
        Args:
            position: Tọa độ ô hiện tại cần ước lượng.
            goal: Tọa độ ô đích Goal.
            
        Returns:
            Khoảng cách ước lượng (float) >= 0.0 từ position đến goal.
        """
        key = (position, goal)
        if key not in self.cache:
            # 1. Chuyển đổi trạng thái ô thành 10 đặc trưng đầu vào
            features = state_to_features(self.maze, position, goal).reshape(1, -1).astype(np.float32)
            
            # 2. Truyền vector đặc trưng vào tensor đầu vào của mô hình TFLite
            self.interpreter.set_tensor(self.input_details[0]['index'], features)
            
            # 3. Kích hoạt thông dịch và suy luận
            self.interpreter.invoke()
            
            # 4. Trích xuất giá trị dự đoán từ tensor đầu ra
            prediction = float(self.interpreter.get_tensor(self.output_details[0]['index'])[0][0])
            
            # Đảm bảo khoảng cách ước lượng không âm và lưu vào cache
            self.cache[key] = max(0.0, prediction)
            
        return self.cache[key]


