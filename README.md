# Maze Pathfinding: A* with Artificial Neural Network (ANN) Heuristic

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue" alt="Python Versions">
  <img src="https://img.shields.io/badge/TensorFlow-2.15+-orange.svg" alt="TensorFlow">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

## 1. Giới thiệu (Introduction)

Dự án này là một minh họa trực quan và chuyên sâu về cách ứng dụng **Học máy (Machine Learning)** vào các bài toán tìm kiếm đồ thị cổ điển. Cụ thể, dự án sử dụng **Mạng nơ-ron nhân tạo (ANN)** để học và xấp xỉ một hàm Heuristic cho thuật toán **A*** trong bài toán tìm đường đi ngắn nhất trong mê cung.

Mục tiêu chính của dự án là so sánh hiệu suất, số lượng node được duyệt và thời gian chạy giữa ba phương pháp:
1. **BFS (Breadth-First Search)**: Tìm kiếm mù, duyệt qua tất cả các trạng thái.
2. **A* + Manhattan Heuristic**: Tìm kiếm heuristic kinh điển, sử dụng khoảng cách Manhattan.
3. **A* + ANN Heuristic**: Tìm kiếm heuristic hiện đại, sử dụng mạng nơ-ron để dự đoán chi phí đường đi dựa trên dữ liệu học được.

Dự án cũng đồng thời là một bài thực hành tuyệt vời để hiểu về các khái niệm trong Machine Learning như: **Overfitting, Underfitting, Dropout, Early Stopping**, và **Cross-validation**.

---

## 2. Cấu trúc Dự án (Project Structure)

```text
maze-ann-a-star/
├── data/                  # Dataset CSV sinh ra từ thuật toán BFS để làm nhãn học
├── models/                # Các file model Keras (.keras) và TensorFlow Lite (.tflite)
├── outputs/               # Biểu đồ loss, metrics, ảnh demo và log kết quả huấn luyện
├── reports/               # Báo cáo và phân tích chuyên sâu
├── slides/                # Tài liệu slide thuyết trình
├── src/                   # Source code chính của dự án
│   ├── config.py          # Cấu hình đường dẫn và hằng số
│   ├── maze.py            # Khởi tạo và thao tác trên lưới mê cung
│   ├── search.py          # Triển khai thuật toán BFS, A* và heuristic Manhattan
│   ├── features.py        # Trích xuất đặc trưng (Feature extraction) từ trạng thái mê cung
│   ├── dataset.py         # Xây dựng dataset với nhãn được đánh bằng BFS
│   ├── generate_dataset.py# Script tự động sinh hàng loạt dataset
│   ├── models.py          # Định nghĩa 3 kiến trúc mạng ANN (Underfit, Overfit, Goodfit)
│   ├── train.py           # Script huấn luyện mô hình
│   ├── cross_validate.py  # Script đánh giá chéo (Cross-validation)
│   ├── heuristics.py      # Tích hợp model TensorFlow Lite thành hàm heuristic cho A*
│   ├── visualize.py       # Render hình ảnh mê cung và vẽ đường đi
│   └── demo.py            # Chạy so sánh thực tế giữa BFS, A* Manhattan và A* ANN
├── requirements.txt       # Các thư viện phụ thuộc
└── README.md              # File tài liệu bạn đang đọc
```

---

## 3. Cơ sở Lý thuyết (Theoretical Background)

### 3.1. Thuật toán A* (A* Algorithm)
A* kết hợp ưu điểm của thuật toán Dijkstra và Greedy Best-First Search. Tại mỗi bước, A* chọn đỉnh $n$ để mở rộng dựa trên hàm đánh giá:

$$ f(n) = g(n) + h(n) $$

Trong đó:
- $g(n)$: Chi phí chính xác từ điểm xuất phát (Start) đến node $n$.
- $h(n)$: **Heuristic** - ước lượng chi phí từ node $n$ đến đích (Goal).

### 3.2. Heuristic Manhattan
Trên lưới ô vuông 4 hướng (lên, xuống, trái, phải) không có vật cản đường chéo, khoảng cách Manhattan là một heuristic **admissible** (luôn ước lượng thấp hơn hoặc bằng thực tế).
Công thức: $h(n) = |x_{current} - x_{goal}| + |y_{current} - y_{goal}|$

### 3.3. Heuristic học bằng ANN
Thay vì dùng công thức toán học cố định, ta huấn luyện một mạng nơ-ron nhận đầu vào là trạng thái của mê cung và dự đoán khoảng cách đến đích. Mạng nơ-ron có thể nắm bắt được sự tồn tại của các bức tường (vật cản) xung quanh để đưa ra dự đoán thực tế hơn so với Manhattan.

---

## 4. Dữ liệu (Dataset)

Để mạng nơ-ron học được khoảng cách, chúng ta cần một tập dữ liệu (dataset) có gán nhãn. Quá trình tạo dữ liệu diễn ra như sau:
1. **Tạo mê cung ngẫu nhiên**: Sinh các lưới với các vật cản (tường) phân bố ngẫu nhiên.
2. **Trích xuất đặc trưng (Feature Extraction)**: Mỗi trạng thái được chuyển thành một vector gồm 10 chiều:
   - Tọa độ hiện tại: `x`, `y`
   - Tọa độ đích: `goal_x`, `goal_y`
   - Vector hướng: `dx = goal_x - x`, `dy = goal_y - y`
   - Cảm biến tường xung quanh (4 chiều): `up`, `down`, `left`, `right` (giá trị 0 hoặc 1).
3. **Gán nhãn (Labeling)**: Chạy thuật toán **BFS** từ trạng thái hiện tại đến đích để tìm khoảng cách ngắn nhất thực tế (`true_distance`). Khoảng cách này được dùng làm nhãn (target) để huấn luyện ANN.

---

## 5. Mô hình Học Máy (Machine Learning Models)

Dự án thiết kế sẵn 3 kiến trúc mô hình khác nhau để so sánh và làm rõ các hiện tượng trong quá trình huấn luyện:

| Tên Thí Nghiệm | Kiến trúc Mạng (Các lớp Dense) | Số Epoch | Dữ liệu Train | Kỹ thuật dùng | Mục đích Minh họa |
| :--- | :--- | :---: | :---: | :--- | :--- |
| **Underfit** | Nhỏ (Input $\to$ 4 $\to$ 1) | 5 | Toàn bộ | Không có | Mô hình quá đơn giản, học ít epoch $\Rightarrow$ Không nắm bắt được quy luật. |
| **Overfit** | Rất Lớn (Input $\to$ 512 $\to$ 512 $\to$ 256 $\to$ 128 $\to$ 1) | 150 | Giới hạn (3,000) | Không có | Mô hình phức tạp, dữ liệu ít, train lâu $\Rightarrow$ "Học vẹt" dữ liệu train, dự đoán kém trên test. |
| **Goodfit** | Vừa phải (Input $\to$ 64 $\to$ Dropout(0.2) $\to$ 32 $\to$ 16 $\to$ 1) | 100 | Toàn bộ | Dropout, Early Stopping | Cân bằng tốt, chống overfit, dừng sớm khi validation loss không giảm. |

*Hàm loss: Mean Squared Error (MSE) - Metric: Mean Absolute Error (MAE)*

---

## 6. Hướng dẫn Cài đặt & Sử dụng

### 6.1. Cài đặt Môi trường
Khuyến nghị sử dụng Python **3.10, 3.11, hoặc 3.12**.

```bash
# 1. Tạo môi trường ảo
python -m venv .venv
source .venv/bin/activate  # (Với Windows: .venv\Scripts\activate)

# 2. Cập nhật pip và cài đặt thư viện
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 6.2. Pipeline Chạy Dự án

**Bước 1: Sinh dữ liệu huấn luyện (Generate Dataset)**
Sinh ngẫu nhiên 500 mê cung, mỗi mê cung lấy tối đa 200 mẫu. File đầu ra: `data/maze_dataset.csv`.
```bash
python -m src.generate_dataset --mazes 500 --max-samples-per-maze 200
```

**Bước 2: Huấn luyện các Mô hình ANN (Train Models)**
Chạy cả 3 thí nghiệm (underfit, overfit, goodfit). File model sẽ lưu tại thư mục `models/`, lịch sử và logs lưu tại `outputs/`.
```bash
python -m src.train --experiment all
```

**Bước 3: Chạy Đánh giá chéo (Cross-Validation)**
Để kiểm chứng độ ổn định của model, chạy 5-fold cross-validation:
```bash
python -m src.cross_validate --folds 5 --epochs 30
```

**Bước 4: Chạy Demo so sánh các thuật toán (Demo)**
Xem trực quan đường đi của BFS, A* Manhattan và A* ANN trên cùng một mê cung.
```bash
python -m src.demo
```
Kết quả sẽ xuất ra terminal dạng bảng metrics và lưu hình ảnh vào `outputs/demo_path.png`.

---

## 7. Kết quả & Đánh giá (Results & Evaluation)

### 7.1. Phân tích quá trình Huấn luyện (Training Analysis)

Qua biểu đồ Loss và Metric MAE, ta có thể thấy rõ sự khác biệt của 3 cách tiếp cận:

<div align="center">
  <table style="text-align:center;">
    <tr>
      <td><b>Underfit</b></td>
      <td><b>Overfit</b></td>
      <td><b>Goodfit</b></td>
    </tr>
    <tr>
      <td><img src="outputs/underfit_loss.png" alt="Underfit" width="250"/></td>
      <td><img src="outputs/overfit_loss.png" alt="Overfit" width="250"/></td>
      <td><img src="outputs/goodfit_loss.png" alt="Goodfit" width="250"/></td>
    </tr>
    <tr>
      <td><i>Loss giảm ít, MAE dừng ở mức cao (~2.6). Mô hình không thể học biểu diễn hàm học phức tạp của mê cung.</i></td>
      <td><i>Train Loss giảm gần bằng 0 nhưng Val Loss tăng vọt. Mô hình "ghi nhớ" thuộc lòng tập train, sai bét trên dữ liệu mới.</i></td>
      <td><i>Train và Val Loss cùng giảm đều và hội tụ sát nhau (MAE ~2.2). Dấu hiệu của một mô hình học khái quát tốt.</i></td>
    </tr>
  </table>
</div>

### 7.2. Kết quả Cross-Validation (Mô hình Goodfit)
*Trung bình sau 5 folds:*
- **MAE**: ~2.283 (Sai số trung bình khoảng 2.2 bước đi)
- **MSE**: ~14.704

### 7.3. So sánh Thuật toán (Demo Path)

Kết quả khi đưa 3 thuật toán vào chạy chung một màn chơi:
- **Khởi điểm (Start)**: `(17, 1)`
- **Đích đến (Goal)**: `(7, 7)`

| Thuật toán | Tìm thấy đường | Chiều dài đường đi | Số Node Mở Rộng | Thời gian chạy |
| :--- | :---: | :---: | :---: | :---: |
| **BFS** | Có | 20 | 104 | ~0.09 ms |
| **A* + Manhattan** | Có | 20 | 46 | ~0.06 ms |
| **A* + ANN (Goodfit)** | Có | 20 | **34** | ~0.37 ms* |

*(Ghi chú: Thời gian của ANN được tối ưu hóa cực tốt nhờ sử dụng định dạng TensorFlow Lite (TFLite) thay vì Keras, loại bỏ phần lớn overhead khi suy luận trên CPU.)*

**Hình ảnh Demo kết quả tìm đường:**

<p align="center">
  <img src="outputs/demo_path.png" alt="Demo Path Comparison" width="600"/>
</p>
<p align="center"><i>(Đường màu xanh mô tả lộ trình tìm được. Số lượng ô màu nhạt xung quanh biểu thị số lượng Node đã bị thuật toán mở rộng (explore))</i></p>

**Nhận xét:**
- **BFS** duyệt một lượng lớn node vì nó tìm kiếm mù theo mọi hướng.
- **A* Manhattan** định hướng tốt hơn, giảm hơn 50% số node so với BFS.
- **A* ANN** cho hiệu quả định hướng cực tốt, **số node cần duyệt là ít nhất**. Hàm Heuristic ANN đã nhận biết được các bức tường từ xa và dẫn đường đi hiệu quả.

---

## 8. Ứng dụng & Hướng Phát Triển (Future Work)

**Ý nghĩa:** Dự án này là công cụ giảng dạy/học tập cực tốt để giải thích vì sao cần Machine Learning, tác hại của Underfit/Overfit, và vai trò của Heuristic trong Trí tuệ nhân tạo (AI).

**Gợi ý phát triển tiếp:**
- Huấn luyện với mê cung kích thước đa dạng và phức tạp hơn (ví dụ Maze sinh bởi thuật toán DFS, Prim).
- Áp dụng mạng **CNN (Convolutional Neural Network)** nhận đầu vào trực tiếp là ảnh/ma trận mê cung 2D cục bộ thay vì vector 1D thủ công.
- Export mô hình sang `.tflite` (đã áp dụng) hoặc `ONNX` để tăng tốc độ inference.
- So sánh thêm với biến thể **Weighted A***.
- Xây dựng giao diện Web/GUI (Pygame/Streamlit) để người dùng có thể tự vẽ tường và xem thuật toán chạy trực tiếp.

---
*Dự án thực hiện nhằm mục đích học tập & nghiên cứu thuật toán AI cơ bản. Chúc bạn có những trải nghiệm thú vị khi khám phá kết hợp giữa Search Algorithms và Machine Learning!*
