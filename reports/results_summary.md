# Tóm tắt kết quả chạy thực nghiệm

Ngày cập nhật kết quả: 2026-07-09

## Dataset

- File: `data/maze_dataset.csv`
- Số dòng: 100,000
- Số maze: 500
- Kích thước maze: 20x20
- Mỗi maze lấy tối đa 200 state reachable
- Nhãn `true_distance` được tạo bằng BFS từ goal
- Input ANN gồm 10 đặc trưng: `x`, `y`, `goal_x`, `goal_y`, `dx`, `dy`, `up_wall`, `down_wall`, `left_wall`, `right_wall`
- Output ANN: khoảng cách ngắn nhất thật từ state hiện tại đến goal

## Kết quả train ANN

| Mô hình | Train rows | Validation rows | Test rows | Epochs ran | Test MSE | Test MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Underfit | 70,000 | 15,000 | 15,000 | 5 | 16.322 | 2.669 |
| Overfit | 3,000 | 15,000 | 15,000 | 150 | 24.811 | 3.064 |
| Good fit | 70,000 | 15,000 | 15,000 | 20 | 14.300 | 2.284 |

Nhận xét:

- Underfit dùng mạng quá nhỏ và chỉ train 5 epoch, nên sai số còn cao.
- Overfit dùng mạng lớn, train 150 epoch trên 3,000 dòng train; train loss giảm nhưng validation loss tăng mạnh về cuối.
- Good fit dùng kiến trúc vừa phải, Dropout và EarlyStopping, đạt Test MAE thấp nhất khoảng 2.28 bước.

## Cross-validation

Chạy 5-fold cross-validation với 10 epoch/fold:

| Fold | Validation rows | MAE | MSE |
| --- | ---: | ---: | ---: |
| 1 | 20,000 | 2.250 | 15.436 |
| 2 | 20,000 | 2.175 | 15.181 |
| 3 | 20,000 | 2.305 | 14.340 |
| 4 | 20,000 | 2.206 | 14.107 |
| 5 | 20,000 | 2.179 | 15.230 |

- Mean MAE: 2.223
- Mean MSE: 14.859

## Demo BFS, A* Manhattan, A* ANN

- Start: `(17, 1)`
- Goal: `(7, 7)`

| Thuật toán | Tìm thấy đường | Độ dài đường đi | Node đã duyệt | Thời gian |
| --- | --- | ---: | ---: | ---: |
| BFS | Có | 20 | 104 | 0.09 ms |
| A* + Manhattan | Có | 20 | 46 | 0.07 ms |
| A* + ANN | Có | 20 | 33 | 0.48 ms |

Nhận xét:

- Cả 3 thuật toán đều tìm được đường dài 20 bước.
- A* + Manhattan duyệt ít node hơn BFS.
- A* + ANN duyệt ít node nhất trong demo này, giảm khoảng 68.3% so với BFS và 28.3% so với A* + Manhattan.
- Thời gian A* + ANN cao hơn Manhattan do phải gọi mô hình, nhưng vẫn dưới 1 ms trong demo nhờ dùng TensorFlow Lite.

## File kết quả đã tạo

- `data/maze_dataset.csv`
- `models/ann_heuristic_underfit.keras`
- `models/ann_heuristic_overfit.keras`
- `models/ann_heuristic_goodfit.keras`
- `models/ann_heuristic_underfit.tflite`
- `models/ann_heuristic_overfit.tflite`
- `models/ann_heuristic_goodfit.tflite`
- `outputs/underfit_loss.png`
- `outputs/overfit_loss.png`
- `outputs/goodfit_loss.png`
- `outputs/demo_path.png`
- `outputs/training_summary.json`
- `outputs/cross_validation.json`
- `outputs/demo_metrics.json`
