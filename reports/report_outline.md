# Khung báo cáo

## 1. Giới thiệu

- Giới thiệu bài toán tìm đường trong mê cung.
- Lý do chọn kết hợp ANN với A*.
- Mục tiêu: ANN học hàm heuristic dự đoán chi phí còn lại đến đích.

## 2. Cơ sở lý thuyết ANN

- ANN gồm input layer, hidden layer, output layer.
- Forward propagation: dữ liệu đi từ input qua các layer để tạo dự đoán.
- Loss function: đo sai lệch giữa dự đoán và nhãn thật.
- Backpropagation: lan truyền sai số ngược lại để cập nhật trọng số.
- Epoch: một lần mô hình học qua toàn bộ tập train.

## 3. Train / Validation / Test

- Training set: dùng để cập nhật trọng số.
- Validation set: theo dõi quá trình học và chọn cấu hình.
- Testing set: chỉ dùng để đánh giá cuối cùng.
- Tỉ lệ sử dụng trong project: 70% train, 15% validation, 15% test.

## 4. Chỉ số đánh giá

- Accuracy: phù hợp với bài toán phân loại.
- ROC/AUC: đánh giá khả năng phân tách lớp trong bài toán classification.
- Cross-validation: chia dữ liệu thành k fold để đánh giá độ ổn định.
- Trong project này, ANN dự đoán khoảng cách nên là bài toán regression.
- Chỉ số chính: MSE loss và MAE. MAE = sai lệch trung bình bao nhiêu bước.

## 5. Thuật toán A*

- A* tìm đường bằng công thức `f(n) = g(n) + h(n)`.
- `g(n)`: chi phí đã đi từ start đến node hiện tại.
- `h(n)`: chi phí ước lượng từ node hiện tại đến goal.
- Manhattan distance là heuristic truyền thống:

```text
h(n) = |x - goal_x| + |y - goal_y|
```

## 6. Thiết kế bài toán mê cung

- Mê cung là lưới 20x20.
- `0` là đường đi, `1` là tường.
- Agent đi 4 hướng: lên, xuống, trái, phải.
- Mục tiêu: tìm đường từ start đến goal.

## 7. Thiết kế dataset cho ANN

Mỗi mẫu dữ liệu là một trạng thái:

```text
[x, y, goal_x, goal_y, dx, dy, up_wall, down_wall, left_wall, right_wall]
```

Output:

```text
true_distance
```

`true_distance` được tính bằng BFS từ goal đến mọi ô có thể đi được.

## 8. Thiết kế 3 mô hình ANN

### Underfitting

- Mạng rất nhỏ.
- Train 5 epoch.
- Kết quả hiện tại: Test MAE = 2.669.

### Overfitting

- Mạng lớn.
- Train 150 epoch trên tập train bị giới hạn 3,000 dòng.
- Kết quả hiện tại: Test MAE = 3.064.

### Good Fit

- Mạng vừa phải.
- Dùng Dropout và EarlyStopping.
- Kết quả hiện tại: dừng ở 20 epoch, Test MAE = 2.284.

## 9. Kết quả thực nghiệm ANN

Chèn 3 hình:

- `outputs/underfit_loss.png`
- `outputs/overfit_loss.png`
- `outputs/goodfit_loss.png`

| Mô hình | Test MSE | Test MAE | Nhận xét |
| --- | ---: | ---: | --- |
| Underfit | 16.322 | 2.669 | Mạng quá nhỏ và học ít epoch. |
| Overfit | 24.811 | 3.064 | Train loss giảm nhưng validation/test kém. |
| Good fit | 14.300 | 2.284 | Tốt nhất trong ba cấu hình. |

## 10. Cross-validation

- 5 folds, 10 epoch/fold.
- Mean MAE = 2.223.
- Mean MSE = 14.859.

## 11. Tích hợp ANN vào A*

Thay Manhattan bằng ANN:

```text
A* truyền thống: f(n) = g(n) + h_manhattan(n)
A* + ANN:        f(n) = g(n) + h_ANN(n)
```

ANN nhận state hiện tại và goal, sau đó trả về chi phí còn lại dự đoán.

## 12. Kết quả demo so sánh

Chèn hình:

- `outputs/demo_path.png`
- UI tương tác: `src/ui.py`, chạy bằng `python -m src.ui`.

| Thuật toán | Tìm thấy đường | Độ dài đường đi | Node đã duyệt | Thời gian |
| --- | --- | ---: | ---: | ---: |
| BFS | Có | 20 | 104 | 0.09 ms |
| A* + Manhattan | Có | 20 | 46 | 0.07 ms |
| A* + ANN | Có | 20 | 33 | 0.48 ms |

## 13. Nhận xét và hạn chế

- BFS đảm bảo tìm đường ngắn nhất nhưng duyệt nhiều node.
- A* + Manhattan duyệt ít node hơn BFS.
- A* + ANN có thể giảm node duyệt trong một số trường hợp.
- ANN heuristic không đảm bảo luôn tối ưu vì có thể dự đoán sai.

## 14. Mức độ sử dụng AI và sản phẩm nộp

- AI hỗ trợ tạo khung project, gợi ý cấu trúc code và cách trình bày.
- Nhóm tự chạy code, đọc kết quả, điền số liệu và giải thích kết quả thực nghiệm.
- Sản phẩm gồm báo cáo PDF, slide, source code, UI tương tác, dataset, model, outputs và README.

## 15. Kết luận

- Project thể hiện được cách ANN học từ dữ liệu được gán nhãn bởi thuật toán truyền thống.
- ANN được ứng dụng làm heuristic cho A*.
- Kết quả cho thấy sự giao thoa giữa AI hiện đại và thuật toán tìm kiếm cổ điển.
