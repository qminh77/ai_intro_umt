# Báo cáo dự thảo

## Đề tài

**Ứng dụng mạng nơ-ron nhân tạo làm hàm heuristic cho thuật toán A* trong bài toán tìm đường mê cung**

## 1. Giới thiệu

Tìm đường trong mê cung là một bài toán quen thuộc của trí tuệ nhân tạo, trong đó tác nhân cần tìm một đường đi từ điểm bắt đầu đến điểm đích mà không đi xuyên qua vật cản. Các thuật toán tìm kiếm truyền thống như BFS và A* có thể giải quyết bài toán này hiệu quả trong nhiều trường hợp.

Trong đề tài này, nhóm kết hợp thuật toán tìm kiếm truyền thống với mạng nơ-ron nhân tạo. Thay vì chỉ dùng heuristic Manhattan trong A*, nhóm huấn luyện một ANN để dự đoán chi phí còn lại từ vị trí hiện tại đến đích. Giá trị dự đoán này sau đó được dùng như hàm heuristic cho A*.

Mục tiêu của đề tài gồm ba phần:

- Hiểu cách ANN học từ dữ liệu, cách chia train/validation/test và cách đọc đồ thị loss.
- Minh họa ba hiện tượng underfitting, overfitting và good fit.
- Tích hợp ANN vào A* để so sánh với BFS và A* dùng Manhattan.

## 2. Cơ sở lý thuyết về ANN

Mạng nơ-ron nhân tạo gồm nhiều lớp tính toán. Lớp đầu vào nhận đặc trưng của dữ liệu, các lớp ẩn học quan hệ phi tuyến, và lớp đầu ra tạo dự đoán cuối cùng. Trong bài toán này, đầu vào của ANN là thông tin về vị trí hiện tại, vị trí đích và vật cản xung quanh; đầu ra là khoảng cách còn lại đến đích.

Quá trình học của ANN gồm hai bước chính. Ở bước lan truyền tiến, dữ liệu đi qua các lớp để tạo ra dự đoán. Sau đó, mô hình so sánh dự đoán với nhãn thật bằng hàm mất mát. Ở bước lan truyền ngược, sai số được truyền ngược qua mạng để điều chỉnh trọng số. Quá trình này lặp lại qua nhiều epoch.

Một epoch là một lần mô hình học qua toàn bộ tập train. Nếu số epoch quá ít hoặc mô hình quá nhỏ, mô hình có thể chưa học đủ. Nếu mô hình quá lớn và học quá lâu trên dữ liệu hạn chế, mô hình có thể học quá kỹ dữ liệu train và giảm khả năng tổng quát trên dữ liệu mới.

## 3. Train, Validation và Test

Dữ liệu được chia thành ba phần:

- Training set: dùng để cập nhật trọng số của mô hình.
- Validation set: dùng để theo dõi quá trình học và điều chỉnh cấu hình.
- Testing set: dùng để đánh giá cuối cùng sau khi mô hình đã được chọn.

Trong project, dữ liệu được chia theo tỉ lệ 70/15/15. Việc tách validation khỏi test là quan trọng vì validation được dùng trong quá trình thử nghiệm cấu hình, còn test chỉ nên được dùng một lần để đánh giá khách quan.

## 4. Chỉ số đánh giá

Với bài toán phân loại, các chỉ số thường dùng là accuracy, ROC/AUC và confusion matrix. Accuracy cho biết tỉ lệ dự đoán đúng. ROC/AUC đánh giá khả năng phân biệt giữa các lớp ở nhiều ngưỡng quyết định khác nhau.

Trong project này, đầu ra của ANN là một số thực biểu diễn khoảng cách còn lại, nên đây là bài toán hồi quy. Vì vậy, nhóm dùng các chỉ số:

- MSE: bình phương sai số trung bình, phạt mạnh các dự đoán sai nhiều.
- MAE: sai số tuyệt đối trung bình, dễ hiểu vì có đơn vị là số bước.

Ví dụ, MAE = 2.27 nghĩa là mô hình dự đoán sai trung bình khoảng 2.27 bước.

Cross-validation cũng được dùng để kiểm tra độ ổn định của mô hình. Dữ liệu được chia thành nhiều fold; mỗi lần một fold làm validation và các fold còn lại làm train. Kết quả trung bình qua các fold giúp đánh giá mô hình ít phụ thuộc hơn vào một lần chia dữ liệu cụ thể.

## 5. Thuật toán BFS và A*

BFS duyệt các trạng thái theo từng lớp khoảng cách. Trong mê cung không trọng số, BFS đảm bảo tìm được đường ngắn nhất nếu đường tồn tại. Tuy nhiên, BFS không dùng thông tin hướng đến đích nên có thể duyệt nhiều node.

A* cải thiện bằng cách dùng hàm đánh giá:

```text
f(n) = g(n) + h(n)
```

Trong đó:

- `g(n)` là chi phí đã đi từ start đến node hiện tại.
- `h(n)` là chi phí ước lượng từ node hiện tại đến goal.
- `f(n)` là tổng chi phí dự đoán.

Heuristic Manhattan được tính như sau:

```text
h(n) = |x - goal_x| + |y - goal_y|
```

Trong mê cung 4 hướng không trọng số, Manhattan là một heuristic đơn giản và thường hiệu quả. Tuy nhiên, Manhattan không nhìn thấy cấu trúc vật cản, nên nhóm thử dùng ANN để học một heuristic từ dữ liệu.

## 6. Thiết kế mê cung và dữ liệu

Mê cung được biểu diễn bằng ma trận 20x20:

```text
0 = đường đi
1 = tường / vật cản
```

Tác nhân có thể đi bốn hướng: lên, xuống, trái, phải. Với mỗi mê cung, nhóm chọn một điểm đích và dùng BFS để tính khoảng cách ngắn nhất thật từ mọi ô có thể đi được đến đích. Nhờ vậy, dữ liệu huấn luyện không cần gán nhãn thủ công.

Mỗi mẫu dữ liệu có 10 đặc trưng:

```text
x, y, goal_x, goal_y, dx, dy, up_wall, down_wall, left_wall, right_wall
```

Trong đó:

- `x, y` là vị trí hiện tại.
- `goal_x, goal_y` là vị trí đích.
- `dx, dy` là độ lệch theo trục x và y.
- `up_wall, down_wall, left_wall, right_wall` cho biết bốn ô xung quanh có phải tường hoặc ra ngoài biên không.

Đầu ra của mô hình là:

```text
true_distance
```

Đây là số bước ngắn nhất thật từ ô hiện tại đến đích.

## 7. Thiết kế mô hình ANN

Nhóm xây dựng ba cấu hình ANN để minh họa ba trạng thái học.

### 7.1. Underfitting

Mô hình underfit có kiến trúc rất nhỏ: một hidden layer với 4 neuron và chỉ train 5 epoch. Mục tiêu là làm cho mô hình không đủ năng lực học quan hệ trong dữ liệu.

### 7.2. Overfitting

Mô hình overfit có nhiều layer và nhiều neuron hơn. Nhóm cố tình train 150 epoch trên một phần nhỏ dữ liệu train. Mục tiêu là tạo tình huống mô hình học quá kỹ tập train, trong khi validation loss không cải thiện tương ứng.

### 7.3. Good fit

Mô hình good fit dùng kiến trúc vừa phải với các layer 64, 32 và 16 neuron. Nhóm thêm Dropout và EarlyStopping để giảm overfitting. EarlyStopping dừng train khi validation loss không cải thiện sau một số epoch.

## 8. Kết quả huấn luyện

Dataset có 100,000 dòng, được sinh từ 500 mê cung. Kết quả test:

| Mô hình | Train rows | Validation rows | Test rows | Epochs | Test MSE | Test MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Underfit | 70,000 | 15,000 | 15,000 | 5 | 16.253 | 2.619 |
| Overfit | 3,000 | 15,000 | 15,000 | 150 | 22.276 | 2.747 |
| Good fit | 70,000 | 15,000 | 15,000 | 15 | 14.416 | 2.268 |

Kết quả cho thấy mô hình good fit có MAE thấp nhất. Mô hình overfit tuy train lâu và lớn hơn nhưng test MAE lại kém hơn, cho thấy mô hình không tổng quát tốt bằng cấu hình có kiểm soát.

Cần chèn các hình sau vào báo cáo:

- `outputs/underfit_loss.png`
- `outputs/overfit_loss.png`
- `outputs/goodfit_loss.png`

## 9. Kết quả cross-validation

Nhóm chạy 5-fold cross-validation với 10 epoch mỗi fold:

| Fold | Validation rows | MAE | MSE |
| --- | ---: | ---: | ---: |
| 1 | 20,000 | 2.245 | 15.241 |
| 2 | 20,000 | 2.292 | 15.139 |
| 3 | 20,000 | 2.208 | 14.751 |
| 4 | 20,000 | 2.206 | 14.018 |
| 5 | 20,000 | 2.465 | 14.373 |

MAE trung bình là 2.283. Kết quả này cho thấy mô hình có sai số tương đối ổn định qua nhiều cách chia dữ liệu.

## 10. Tích hợp ANN vào A*

Sau khi huấn luyện, mô hình good fit được lưu lại và dùng làm heuristic cho A*. Khi A* cần đánh giá một node, hệ thống chuyển trạng thái đó thành vector 10 đặc trưng rồi gọi ANN để dự đoán chi phí còn lại.

So sánh:

```text
A* truyền thống: f(n) = g(n) + h_manhattan(n)
A* + ANN:        f(n) = g(n) + h_ANN(n)
```

ANN heuristic không đảm bảo luôn tối ưu như một heuristic admissible. Vì ANN có thể dự đoán sai hoặc dự đoán lớn hơn khoảng cách thật, đường đi tìm được không được đảm bảo tối ưu tuyệt đối trong mọi trường hợp. Do đó, nhóm đánh giá bằng thực nghiệm thay vì khẳng định ANN luôn tốt hơn.

## 11. Kết quả demo

Demo dùng cùng một mê cung cho ba thuật toán:

- BFS
- A* + Manhattan
- A* + ANN

Kết quả:

| Thuật toán | Tìm thấy đường | Độ dài đường đi | Node đã duyệt | Thời gian |
| --- | --- | ---: | ---: | ---: |
| BFS | Có | 20 | 104 | 0.10 ms |
| A* + Manhattan | Có | 20 | 46 | 0.07 ms |
| A* + ANN | Có | 20 | 33 | 680.28 ms |

Trong lần demo này, cả ba thuật toán đều tìm được đường dài 20 bước. A* + Manhattan duyệt ít node hơn BFS. A* + ANN duyệt ít node nhất, cho thấy heuristic học được có thể hướng tìm kiếm tốt hơn trong trường hợp này.

Tuy nhiên, thời gian chạy của A* + ANN cao hơn do mỗi lần tính heuristic phải gọi mô hình Keras. Đây là overhead của demo Python, không nhất thiết phản ánh chất lượng heuristic. Khi trình bày, nhóm nên nhấn mạnh tiêu chí số node đã duyệt và tính thực nghiệm của kết quả.

Cần chèn hình:

- `outputs/demo_path.png`

## 12. Nhận xét và hạn chế

Ưu điểm:

- Dataset được tạo tự động bằng BFS, không cần gán nhãn thủ công.
- Project minh họa rõ underfitting, overfitting và good fit.
- ANN được dùng trong một bài toán thuật toán cụ thể, không chỉ train mô hình độc lập.

Hạn chế:

- ANN chỉ nhìn bốn ô xung quanh nên chưa hiểu toàn bộ cấu trúc mê cung.
- ANN heuristic không đảm bảo tối ưu tuyệt đối.
- Gọi model Keras nhiều lần trong A* làm demo chạy chậm hơn.

Hướng phát triển:

- Thêm đặc trưng vùng lân cận lớn hơn, ví dụ cửa sổ 3x3 hoặc 5x5.
- Train trên nhiều mê cung hơn.
- Batch prediction hoặc cache nhiều trạng thái để giảm thời gian gọi model.
- So sánh thêm với Greedy Best-First Search.

## 13. Kết luận

Đề tài đã xây dựng được một hệ thống hoàn chỉnh gồm sinh mê cung, tạo dataset bằng BFS, huấn luyện ANN, minh họa underfitting/overfitting/good fit và tích hợp ANN vào A*. Kết quả demo cho thấy ANN có thể đóng vai trò heuristic học từ dữ liệu và giúp A* giảm số node duyệt trong một số trường hợp.

Tuy nhiên, ANN không thay thế hoàn toàn heuristic truyền thống vì không đảm bảo tính tối ưu. Kết quả phù hợp nhất nên được hiểu là một minh họa cho sự kết hợp giữa AI hiện đại và thuật toán tìm kiếm cổ điển.

## 14. Mức độ sử dụng AI hỗ trợ

AI được dùng để hỗ trợ:

- Gợi ý cấu trúc project.
- Sinh khung code ban đầu.
- Hỗ trợ debug và tổ chức báo cáo.

Nhóm cần tự chạy lại code, đọc kết quả thật, chỉnh sửa nội dung báo cáo/slide bằng lời của mình và hiểu rõ từng phần code để trả lời phản biện.

