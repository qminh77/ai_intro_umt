# Nội dung slide dự thảo

## Slide 1: Tên đề tài

Ứng dụng mạng nơ-ron nhân tạo làm hàm heuristic cho thuật toán A* trong bài toán tìm đường mê cung.

## Slide 2: Lý do chọn đề tài

- Mê cung là bài toán trực quan, dễ demo.
- A* là thuật toán tìm kiếm cổ điển quan trọng.
- ANN giúp thử nghiệm ý tưởng học heuristic từ dữ liệu thay vì thiết kế hoàn toàn bằng tay.

## Slide 3: Bài toán

- Mê cung 20x20.
- `0` là đường đi, `1` là tường.
- Tác nhân đi bốn hướng.
- Mục tiêu: tìm đường từ Start đến Goal.

## Slide 4: BFS

- BFS duyệt theo từng lớp khoảng cách.
- Đảm bảo đường ngắn nhất trong mê cung không trọng số.
- Hạn chế: có thể duyệt nhiều node vì không dùng hướng đến goal.

## Slide 5: A* và heuristic

```text
f(n) = g(n) + h(n)
```

- `g(n)`: chi phí đã đi.
- `h(n)`: chi phí còn lại được ước lượng.
- Heuristic càng tốt, A* càng có thể duyệt ít node.

## Slide 6: Manhattan heuristic

```text
h(n) = |x - goal_x| + |y - goal_y|
```

- Đơn giản, nhanh.
- Phù hợp với lưới 4 hướng.
- Nhưng không nhìn trực tiếp cấu trúc vật cản.

## Slide 7: Ý tưởng ANN heuristic

- Dùng ANN để dự đoán khoảng cách còn lại đến goal.
- ANN nhận thông tin vị trí và tường xung quanh.
- Giá trị ANN dự đoán được dùng làm `h(n)` trong A*.

## Slide 8: ANN học như thế nào

- Forward: dữ liệu đi qua mạng để tạo dự đoán.
- Loss: đo sai số giữa dự đoán và nhãn thật.
- Backpropagation: cập nhật trọng số để giảm loss.
- Epoch: một lần học qua toàn bộ tập train.

## Slide 9: Train / Validation / Test

- Train: cập nhật trọng số.
- Validation: theo dõi quá trình học và chọn cấu hình.
- Test: đánh giá cuối cùng.
- Project chia 70/15/15.

## Slide 10: Dataset

Input gồm 10 đặc trưng:

```text
x, y, goal_x, goal_y, dx, dy,
up_wall, down_wall, left_wall, right_wall
```

Output:

```text
true_distance
```

Nhãn `true_distance` được tạo bằng BFS.

## Slide 11: Underfitting

- Mạng quá nhỏ.
- Chỉ train 5 epoch.
- Kết quả: Test MAE = 2.619.
- Chèn hình `outputs/underfit_loss.png`.

## Slide 12: Overfitting

- Mạng lớn.
- Train 150 epoch trên 3,000 dòng train.
- Train loss giảm nhưng validation loss tăng/dị động.
- Test MAE = 2.747.
- Chèn hình `outputs/overfit_loss.png`.

## Slide 13: Good Fit

- Mạng vừa phải.
- Có Dropout và EarlyStopping.
- Dừng ở 15 epoch.
- Test MAE = 2.268.
- Chèn hình `outputs/goodfit_loss.png`.

## Slide 14: Demo so sánh

Chèn hình `outputs/demo_path.png`.

| Thuật toán | Độ dài | Node duyệt |
| --- | ---: | ---: |
| BFS | 20 | 104 |
| A* + Manhattan | 20 | 46 |
| A* + ANN | 20 | 33 |

## Slide 15: Kết luận

- ANN có thể học heuristic từ dữ liệu sinh bằng BFS.
- A* + ANN duyệt ít node nhất trong demo.
- ANN heuristic không đảm bảo luôn tối ưu vì có thể dự đoán sai.
- Đề tài thể hiện sự kết hợp giữa AI hiện đại và thuật toán tìm kiếm truyền thống.

