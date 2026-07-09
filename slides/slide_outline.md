# Khung slide thuyết trình 15-20 phút

## Slide 1: Tên đề tài

- Ứng dụng ANN làm heuristic cho A* trong bài toán tìm đường mê cung.
- Thành viên nhóm.

## Slide 2: Mục tiêu đề tài

- ANN học như thế nào.
- Minh họa underfit, overfit, good fit.
- Tích hợp ANN vào A*.

## Slide 3: Bài toán tổng quan

- Mê cung 20x20.
- Start, Goal, tường và đường đi.
- Agent đi 4 hướng.

## Slide 4: BFS

- Duyệt theo từng lớp khoảng cách.
- Đảm bảo đường ngắn nhất.
- Dùng để tạo nhãn `true_distance`.

## Slide 5: A* hoạt động như thế nào

- `f(n) = g(n) + h(n)`.
- Vai trò của heuristic.

## Slide 6: Manhattan heuristic

- Công thức Manhattan.
- Điểm mạnh: đơn giản, nhanh.
- Hạn chế: không học từ dữ liệu.

## Slide 7: Ý tưởng thay heuristic bằng ANN

- ANN nhận state của ô hiện tại.
- ANN dự đoán chi phí còn lại đến Goal.
- A* dùng giá trị này làm `h(n)`.

## Slide 8: ANN học như thế nào

- Forward propagation.
- Loss.
- Backpropagation.
- Epoch.

## Slide 9: Train / Validation / Test

- Training: học trọng số.
- Validation: theo dõi và chỉnh cấu hình.
- Test: đánh giá cuối cùng.
- Tỉ lệ 70/15/15.

## Slide 10: Chỉ số đánh giá

- Accuracy và ROC/AUC dùng cho classification.
- Project dùng regression nên dùng MSE và MAE.
- Cross-validation kiểm tra độ ổn định.

## Slide 11: Input / Output của ANN

Input:

```text
x, y, goal_x, goal_y, dx, dy, up_wall, down_wall, left_wall, right_wall
```

Output:

```text
true_distance
```

## Slide 12: Underfitting

- Mô hình quá nhỏ.
- Train 5 epoch.
- Test MAE = 2.669.
- Chèn hình `outputs/underfit_loss.png`.

## Slide 13: Overfitting

- Mô hình quá lớn.
- Train 150 epoch trên 3,000 dòng.
- Test MAE = 3.064.
- Chèn hình `outputs/overfit_loss.png`.

## Slide 14: Good Fit

- Mô hình vừa phải.
- Dropout và EarlyStopping.
- Dừng ở 20 epoch.
- Test MAE = 2.284.
- Chèn hình `outputs/goodfit_loss.png`.

## Slide 15: Cross-validation

- 5 folds, 10 epoch/fold.
- Mean MAE = 2.223.
- Mean MSE = 14.859.

## Slide 16: Demo so sánh BFS, A* Manhattan, A* ANN

- Chèn `outputs/demo_path.png`.
- Có UI tương tác tại `src/ui.py`, chạy bằng `python -m src.ui`.
- Khi bấm chạy thuật toán, UI tự animate node đã duyệt, đánh số thứ tự và vẽ 3 path bằng 3 màu thuật toán.
- BFS: 104 nodes.
- A* Manhattan: 46 nodes.
- A* ANN: 33 nodes.

## Slide 17: Nhận xét và hạn chế

- A* + ANN giảm node duyệt trong demo.
- Input còn cục bộ, chưa nhìn toàn bộ mê cung.

## Slide 18: Kết luận và minh bạch AI

- Đề tài thể hiện sự kết hợp giữa ANN huấn luyện bằng Keras và thuật toán A*.
- AI hỗ trợ khung code/debug/tổ chức nội dung.
- Nhóm tự chạy code, đọc kết quả và giải thích cấu hình.
