# Nội dung slide hoàn chỉnh

## Slide 1: Tên đề tài

Ứng dụng mạng nơ-ron nhân tạo làm hàm heuristic cho thuật toán A* trong bài toán tìm đường mê cung.

Nhóm Tờ Linh: Nguyễn Quốc Minh, Võ Nguyễn Tấn Tài.

## Slide 2: Mục tiêu đề tài

- Hiểu ANN học như thế nào qua forward, loss, backpropagation và epoch.
- Phân biệt training, validation và testing set.
- Minh họa underfitting, overfitting và good fit bằng đồ thị train thật.
- Tích hợp ANN vào A* để thay thế heuristic Manhattan trong demo mê cung.

## Slide 3: Bài toán mê cung

- Mê cung 20x20.
- `0` là đường đi, `1` là tường.
- Tác nhân đi bốn hướng: lên, xuống, trái, phải.
- Mục tiêu: tìm đường từ Start đến Goal với số bước hợp lý.

## Slide 4: BFS

- BFS duyệt theo từng lớp khoảng cách.
- Đảm bảo tìm đường ngắn nhất trong mê cung không trọng số.
- Hạn chế: có thể duyệt nhiều node vì không dùng hướng đến goal.
- Trong project, BFS còn dùng để tạo nhãn `true_distance` cho ANN.

## Slide 5: A* và heuristic

```text
f(n) = g(n) + h(n)
```

- `g(n)`: chi phí đã đi.
- `h(n)`: chi phí còn lại được ước lượng.
- Heuristic càng gần khoảng cách thật, A* thường càng duyệt ít node.

## Slide 6: Manhattan heuristic

```text
h(n) = |x - goal_x| + |y - goal_y|
```

- Đơn giản, nhanh.
- Phù hợp với lưới 4 hướng.
- Không trực tiếp học từ dữ liệu và không nhìn đầy đủ vật cản phức tạp.

## Slide 7: Ý tưởng ANN heuristic

- Dùng ANN để dự đoán khoảng cách còn lại đến goal.
- ANN nhận thông tin vị trí hiện tại, vị trí goal và tường lân cận.
- Giá trị ANN dự đoán được dùng làm `h(n)` trong A*.
- Đây là điểm giao thoa giữa học máy hiện đại và thuật toán tìm kiếm truyền thống.

## Slide 8: ANN học như thế nào

- Forward propagation: dữ liệu đi qua mạng để tạo dự đoán.
- Loss: đo sai số giữa dự đoán và nhãn thật.
- Backpropagation: truyền sai số ngược lại để cập nhật trọng số.
- Epoch: một lần mô hình học qua toàn bộ tập train.

## Slide 9: Train / Validation / Test

- Train: cập nhật trọng số.
- Validation: theo dõi quá trình học và chọn cấu hình.
- Test: đánh giá cuối cùng sau khi chọn mô hình.
- Project chia 70/15/15: 70,000 train, 15,000 validation, 15,000 test.

## Slide 10: Chỉ số đánh giá

- Accuracy và ROC/AUC phù hợp với bài toán phân loại.
- Project này là hồi quy vì ANN dự đoán số bước còn lại.
- Loss dùng MSE, metric chính dùng MAE.
- MAE = 2.284 nghĩa là mô hình sai trung bình khoảng 2.284 bước.
- Cross-validation dùng để kiểm tra độ ổn định qua nhiều cách chia dữ liệu.

## Slide 11: Dataset và input/output

Input gồm 10 đặc trưng:

```text
x, y, goal_x, goal_y, dx, dy,
up_wall, down_wall, left_wall, right_wall
```

Output:

```text
true_distance
```

Nhãn `true_distance` được tạo bằng BFS từ goal.

## Slide 12: Underfitting

- Mạng quá nhỏ: Dense(4) + output.
- Chỉ train 5 epoch.
- Kết quả: Test MSE = 16.322, Test MAE = 2.669.
- Chèn hình `outputs/underfit_loss.png`.

## Slide 13: Overfitting

- Mạng lớn: 512, 512, 256, 128 neuron.
- Train 150 epoch trên 3,000 dòng train.
- Train loss giảm nhưng validation loss tăng mạnh về cuối.
- Kết quả: Test MSE = 24.811, Test MAE = 3.064.
- Chèn hình `outputs/overfit_loss.png`.

## Slide 14: Good Fit

- Mạng vừa phải: 64, Dropout(0.2), 32, 16 neuron.
- Có EarlyStopping theo validation loss.
- Dừng ở 20 epoch.
- Kết quả: Test MSE = 14.300, Test MAE = 2.284.
- Chèn hình `outputs/goodfit_loss.png`.

## Slide 15: Cross-validation

5-fold cross-validation, 10 epoch/fold:

| Fold | MAE | MSE |
| --- | ---: | ---: |
| 1 | 2.250 | 15.436 |
| 2 | 2.175 | 15.181 |
| 3 | 2.305 | 14.340 |
| 4 | 2.206 | 14.107 |
| 5 | 2.179 | 15.230 |

Trung bình: MAE = 2.223, MSE = 14.859.

## Slide 16: Demo so sánh thuật toán

Chèn hình `outputs/demo_path.png`.

Demo tương tác: chạy `python -m src.ui`. Khi bấm chạy thuật toán, UI tự animate node đã duyệt, đánh số thứ tự node và có thể xem path của cả 3 thuật toán bằng 3 màu riêng.

| Thuật toán | Độ dài | Node duyệt | Thời gian |
| --- | ---: | ---: | ---: |
| BFS | 20 | 104 | 0.09 ms |
| A* + Manhattan | 20 | 46 | 0.07 ms |
| A* + ANN | 20 | 33 | 0.48 ms |

## Slide 17: Nhận xét và hạn chế

- A* + ANN duyệt ít node nhất trong demo này.
- ANN chỉ dùng 4 ô lân cận, chưa nhìn toàn bộ cấu trúc mê cung.
- Thời gian inference cao hơn Manhattan nhưng vẫn dưới 1 ms trong demo.

## Slide 18: Kết luận và minh bạch AI

- Map được sinh ngẫu nhiên; BFS dùng để gán nhãn `true_distance` cho ANN.
- Good fit tốt hơn underfit và overfit nhờ cấu hình vừa phải, Dropout và EarlyStopping.
- ANN được xây dựng bằng Keras (`tf.keras`) và tích hợp với thuật toán A*.
- AI được dùng để hỗ trợ khung code/debug/tổ chức báo cáo; nhóm tự chạy lại kết quả, chỉnh cấu hình và giải thích số liệu.
