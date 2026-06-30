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

## 8. Kết quả thực nghiệm đã thu được

Dự án đã chạy thành công các bước huấn luyện và demo. Một số kết quả thực tế được lưu trong thư mục outputs như sau.

### 8.1. Kết quả huấn luyện ANN

| Mô hình | Số dòng train | Số dòng validation | Số dòng test | Epoch chạy | Test MSE | Test MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Underfit | 70,000 | 15,000 | 15,000 | 5 | 16.253 | 2.619 |
| Overfit | 3,000 | 15,000 | 15,000 | 150 | 22.276 | 2.747 |
| Good fit | 70,000 | 15,000 | 15,000 | 15 | 14.416 | 2.268 |

Nhận xét:
- Underfit cho thấy mô hình quá nhỏ và học chưa đủ, nên sai số vẫn còn cao.
- Overfit cho thấy mô hình học rất tốt trên dữ liệu huấn luyện nhưng không tổng quát hóa tốt trên dữ liệu test.
- Good fit cho kết quả tốt nhất, với MAE thấp nhất trên tập test.

### 8.2. Kết quả cross-validation

Đã chạy 5-fold cross-validation với các kết quả sau:

| Fold | Số dòng validation | MAE | MSE |
| --- | ---: | ---: | ---: |
| 1 | 20,000 | 2.245 | 15.241 |
| 2 | 20,000 | 2.292 | 15.139 |
| 3 | 20,000 | 2.208 | 14.751 |
| 4 | 20,000 | 2.206 | 14.018 |
| 5 | 20,000 | 2.465 | 14.373 |

- MAE trung bình: 2.283
- MSE trung bình: 14.704

### 8.3. Kết quả demo so sánh thuật toán

Trong một ví dụ demo cụ thể, các thuật toán được chạy trên cùng một mê cung với:
- Start: $(17, 1)$
- Goal: $(7, 7)$

| Thuật toán | Tìm thấy đường | Độ dài đường đi | Số node đã duyệt | Thời gian |
| --- | --- | ---: | ---: | ---: |
| BFS | Có | 20 | 104 | 0.09 ms |
| A* + Manhattan | Có | 20 | 46 | 0.06 ms |
| A* + ANN | Có | 20 | 33 | 681.70 ms |

Nhận xét quan trọng:
- Ba thuật toán đều tìm được đường đi đúng với cùng độ dài 20 bước.
- A* + Manhattan duyệt ít node hơn BFS.
- A* + ANN duyệt ít node nhất trong ví dụ này, cho thấy heuristic học được có thể giúp giảm số lượng nút mở rộng.
- Tuy nhiên, thời gian chạy của A* + ANN cao hơn do overhead của việc gọi model Keras tại mỗi bước đánh giá heuristic.

### 8.4. Các file kết quả được sinh ra

- outputs/underfit_loss.png
- outputs/overfit_loss.png
- outputs/goodfit_loss.png
- outputs/demo_path.png
- outputs/training_summary.json
- outputs/cross_validation.json
- outputs/demo_metrics.json

## 9. Hình ảnh và biểu đồ minh họa

Dự án đã sinh ra các file hình ảnh trực quan để bạn có thể dùng cho báo cáo, slide hoặc thuyết trình. Các biểu đồ này nằm trong thư mục outputs và có thể mở trực tiếp bằng bất kỳ trình xem ảnh nào.

### 9.1. Biểu đồ huấn luyện cho từng mô hình

- [outputs/underfit_loss.png](outputs/underfit_loss.png): biểu đồ loss và MAE của mô hình underfit. Thường cho thấy train/validation đều khá cao và không ổn định.
- [outputs/overfit_loss.png](outputs/overfit_loss.png): biểu đồ loss và MAE của mô hình overfit. Train loss giảm mạnh, nhưng validation/test không cải thiện tốt như kỳ vọng.
- [outputs/goodfit_loss.png](outputs/goodfit_loss.png): biểu đồ loss và MAE của mô hình goodfit. Đây là mô hình cho kết quả tốt nhất, với đường train và validation gần nhau hơn.

### 9.2. Hình minh họa đường đi trên mê cung

- [outputs/demo_path.png](outputs/demo_path.png): hình ảnh minh họa đường đi tìm được bởi các thuật toán BFS, A* + Manhattan và A* + ANN trên cùng một mê cung.

### 9.3. Cách đọc các biểu đồ

- Trục hoành: số epoch.
- Trục tung: giá trị loss hoặc MAE.
- Nếu đường train giảm liên tục nhưng validation tăng hoặc dao động, mô hình có dấu hiệu overfitting.
- Nếu cả train và validation đều cao, mô hình có thể underfit.
- Nếu train và validation đều giảm và tiến gần nhau, đó là dấu hiệu của mô hình good fit.

## 10. Lưu ý quan trọng

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
