# Maze ANN A* Project

Dự án này minh họa cách sử dụng mạng nơ-ron nhân tạo (ANN) để học một hàm heuristic cho thuật toán A* trong bài toán tìm đường trong mê cung. Mục tiêu là so sánh giữa ba phương pháp tìm đường: BFS, A* với heuristic Manhattan và A* với heuristic học được từ dữ liệu.

## 1. Tổng quan

Mê cung được biểu diễn bằng lưới ô vuông, trong đó mỗi ô có thể là:

- 0: ô trống, có thể đi qua
- 1: tường hoặc vật cản
- S: điểm bắt đầu
- G: điểm đích

Ví dụ về mê cung đơn giản:

```text
S . # . .
. . # . G
. . . . .
# . # . .
. . . . .
```

Trong dự án này, thuật toán A* được cải thiện bằng một heuristic được học bởi ANN thay vì dùng heuristic Manhattan thủ công.

## 2. Ý tưởng chính

Thuật toán A* tìm đường bằng cách mở rộng các nút có giá trị:

$$
 f(n) = g(n) + h(n)
$$

Trong đó:
- $g(n)$ là chi phí từ điểm bắt đầu đến nút hiện tại
- $h(n)$ là ước lượng chi phí còn lại từ nút hiện tại đến đích

### Hai loại heuristic được dùng

1. Heuristic Manhattan
   - Công thức: $|x_1-x_2| + |y_1-y_2|$
   - Nhanh và dễ hiểu
   - Thường là heuristic hợp lệ trong mê cung 4 hướng không có trọng số

2. Heuristic bằng ANN
   - Mạng nơ-ron học trực tiếp từ dữ liệu được tạo bằng BFS
   - Mỗi trạng thái đầu vào gồm các đặc trưng về vị trí, mục tiêu và các bức tường xung quanh
   - Đầu ra là ước lượng khoảng cách còn lại đến đích

## 3. Mô hình học và dữ liệu

### 3.1. Đặc trưng đầu vào

Mỗi trạng thái được chuyển thành một vector 10 đặc trưng gồm:

- tọa độ hiện tại $(x, y)$
- tọa độ đích $(goal_x, goal_y)$
- vector hướng $(dx, dy)$
- các thông tin về tường ở bốn phía: up, down, left, right

### 3.2. Nhãn mục tiêu

Nhãn $true\_distance$ được tạo bằng thuật toán BFS từ mỗi trạng thái đến đích. Đây là khoảng cách ngắn nhất thực tế trong mê cung, nên phù hợp làm mục tiêu học cho ANN.

## 4. Kiến trúc mô hình

Dự án xây dựng ba thí nghiệm mạng nơ-ron khác nhau để minh họa các hiện tượng học máy:

| Thí nghiệm | Mô hình | Mục đích |
| --- | --- | --- |
| Underfit | Mạng nhỏ, ít epoch | Minh họa trường hợp học chưa đủ, chưa nắm được mẫu |
| Overfit | Mạng lớn, huấn luyện dài, dữ liệu hạn chế | Minh họa trường hợp quá khớp dữ liệu huấn luyện |
| Good Fit | Mạng vừa phải, có Dropout và EarlyStopping | Minh họa mô hình học tốt, tổng quát hóa tốt |

Các mô hình được triển khai bằng TensorFlow/Keras với hàm mất mát MSE và metric MAE.

## 5. Cấu trúc thư mục

```text
maze-ann-a-star/
├── data/                  # Dataset CSV sinh ra từ mê cung
├── models/                # Các file model Keras đã huấn luyện
├── outputs/               # Biểu đồ loss, metrics và kết quả demo
├── reports/               # Bản thảo báo cáo và kết quả phân tích
├── slides/                # Nội dung slide thuyết trình
├── src/
│   ├── config.py
│   ├── maze.py            # Tạo mê cung và thao tác trên lưới
│   ├── search.py          # BFS, A*, heuristic Manhattan
│   ├── features.py        # Chuyển state thành vector đặc trưng
│   ├── dataset.py         # Tạo dataset có nhãn bằng BFS
│   ├── generate_dataset.py
│   ├── models.py          # Ba kiến trúc ANN cho 3 thí nghiệm
│   ├── train.py           # Huấn luyện các mô hình
│   ├── cross_validate.py  # Cross-validation
│   ├── heuristics.py      # Tích hợp ANN như heuristic cho A*
│   ├── visualize.py       # Vẽ mê cung và đường đi
│   └── demo.py            # So sánh BFS, A* + Manhattan, A* + ANN
├── requirements.txt
└── README.md
```

## 6. Yêu cầu môi trường

Khuyến nghị dùng Python 3.10, 3.11 hoặc 3.12 để tránh các vấn đề tương thích với TensorFlow.

### Cài đặt

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Nếu máy của bạn đang dùng Python mới hơn và TensorFlow gặp lỗi, hãy chuyển về Python 3.11 hoặc 3.12.

## 7. Quy trình chạy dự án

### Bước 1: Tạo dataset

```bash
python -m src.generate_dataset --mazes 500 --max-samples-per-maze 200
```

Kết quả sẽ tạo ra file:

```text
data/maze_dataset.csv
```

Mỗi dòng trong file CSV chứa một trạng thái mê cung cùng nhãn khoảng cách thực tế.

### Bước 2: Huấn luyện 3 thí nghiệm ANN

```bash
python -m src.train --experiment all
```

Kết quả bao gồm:

```text
models/ann_heuristic_underfit.keras
models/ann_heuristic_overfit.keras
models/ann_heuristic_goodfit.keras
outputs/underfit_history.csv
outputs/overfit_history.csv
outputs/goodfit_history.csv
outputs/training_summary.json
```

### Bước 3: Cross-validation

```bash
python -m src.cross_validate --folds 5 --epochs 30
```

Kết quả sẽ lưu vào:

```text
outputs/cross_validation.json
```

### Bước 4: Chạy demo so sánh thuật toán

```bash
python -m src.demo
```

Kết quả bao gồm:

```text
outputs/demo_path.png
outputs/demo_metrics.json
```

Demo sẽ so sánh:
- BFS
- A* + Manhattan
- A* + ANN (nếu đã có model tốt)

## 8. Kết quả mong đợi

Sau khi chạy xong, bạn sẽ thấy:
- đồ thị loss/MAE cho các thí nghiệm underfit, overfit và goodfit
- số lượng nút duyệt và thời gian chạy của từng thuật toán
- hình ảnh minh họa đường đi tìm được trên mê cung

## 9. Lưu ý quan trọng

Thông tin quan trọng khi trình bày hoặc viết báo cáo là:

> ANN không phải lúc nào cũng tốt hơn heuristic Manhattan. Nó là một heuristic học từ dữ liệu và có thể cải thiện hiệu quả trong một số trường hợp, nhưng không đảm bảo luôn tối ưu hoặc luôn hợp lệ như một heuristic admissible.

Đây là điểm đáng chú ý vì heuristic Manhattan có tính chất hợp lệ trong mê cung 4 hướng không có trọng số, trong khi heuristic do ANN học có thể ước lượng sai hoặc overestimate.

## 10. Ứng dụng và ý nghĩa

Dự án này phù hợp để minh họa các khái niệm sau trong môn nhập môn trí tuệ nhân tạo:
- mạng nơ-ron nhân tạo
- huấn luyện, validation, test
- overfitting và underfitting
- hàm mất mát và metric đánh giá
- cross-validation
- ứng dụng học máy vào thuật toán tìm kiếm cổ điển

## 11. Gợi ý phát triển tiếp

Có thể mở rộng dự án theo các hướng sau:
- dùng nhiều loại mê cung phức tạp hơn
- thêm heuristic học sâu hoặc gradient boosting
- so sánh với thuật toán Weighted A*
- xây dựng giao diện trực quan để xem đường đi từng bước

---

Nếu bạn đang sử dụng dự án cho bài thuyết trình, README này có thể làm nền tảng cho phần giới thiệu, phương pháp, kết quả và nhận xét chuyên sâu.
