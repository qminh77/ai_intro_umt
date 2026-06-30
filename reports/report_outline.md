# Khung bao cao

## 1. Gioi thieu

- Gioi thieu bai toan tim duong trong me cung.
- Ly do chon ket hop ANN voi A*.
- Muc tieu: ANN hoc ham heuristic du doan chi phi con lai den dich.

## 2. Co so ly thuyet ANN

- ANN gom input layer, hidden layer, output layer.
- Forward propagation: du lieu di tu input qua cac layer de tao du doan.
- Loss function: do sai lech giua du doan va nhan that.
- Backpropagation: lan truyen sai so nguoc lai de cap nhat trong so.
- Epoch: mot lan mo hinh hoc qua toan bo tap train.

## 3. Train / Validation / Test

- Training set: dung de cap nhat trong so.
- Validation set: theo doi qua trinh hoc va chon cau hinh.
- Testing set: chi dung de danh gia cuoi cung.
- Ti le su dung trong project: 70% train, 15% validation, 15% test.

## 4. Chi so danh gia

- Accuracy: phu hop voi bai toan phan loai.
- ROC/AUC: danh gia kha nang phan tach lop trong bai toan classification.
- Cross-validation: chia du lieu thanh k fold de danh gia do on dinh.
- Trong project nay, ANN du doan khoang cach nen la bai toan regression.
- Chi so chinh: MSE loss va MAE. MAE = sai lech trung binh bao nhieu buoc.

## 5. Thuat toan A*

- A* tim duong bang cong thuc `f(n) = g(n) + h(n)`.
- `g(n)`: chi phi da di tu start den node hien tai.
- `h(n)`: chi phi uoc luong tu node hien tai den goal.
- Manhattan distance la heuristic truyen thong:

```text
h(n) = |x - goal_x| + |y - goal_y|
```

## 6. Thiet ke bai toan me cung

- Me cung la luoi 20x20.
- `0` la duong di, `1` la tuong.
- Agent di 4 huong: len, xuong, trai, phai.
- Muc tieu: tim duong tu start den goal.

## 7. Thiet ke dataset cho ANN

Moi mau du lieu la mot trang thai:

```text
[x, y, goal_x, goal_y, dx, dy, up_wall, down_wall, left_wall, right_wall]
```

Output:

```text
true_distance
```

`true_distance` duoc tinh bang BFS tu goal den moi o co the di duoc.

## 8. Thiet ke 3 mo hinh ANN

### Underfitting

- Mang rat nho.
- So epoch it.
- Ky vong: train loss va validation loss deu cao.

### Overfitting

- Mang lon.
- Train lau tren tap train bi gioi han.
- Ky vong: train loss thap hon nhieu so voi validation loss.

### Good Fit

- Mang vua phai.
- Dung Dropout va EarlyStopping.
- Ky vong: validation loss on dinh, test MAE thap.

## 9. Ket qua thuc nghiem ANN

Chen 3 hinh:

- `outputs/underfit_loss.png`
- `outputs/overfit_loss.png`
- `outputs/goodfit_loss.png`

Co the lay bang so lieu da chay tu `reports/results_summary.md`.

Bang ket qua de dien sau khi chay:

| Mo hinh | Test MSE | Test MAE | Nhan xet |
| --- | ---: | ---: | --- |
| Underfit | ... | ... | ... |
| Overfit | ... | ... | ... |
| Good fit | ... | ... | ... |

## 10. Tich hop ANN vao A*

Thay Manhattan bang ANN:

```text
A* truyen thong: f(n) = g(n) + h_manhattan(n)
A* + ANN:        f(n) = g(n) + h_ANN(n)
```

ANN nhan state hien tai va goal, sau do tra ve chi phi con lai du doan.

## 11. Ket qua demo so sanh

Chen hinh:

- `outputs/demo_path.png`

Co the lay bang so lieu da chay tu `reports/results_summary.md`.

Bang ket qua de dien sau khi chay:

| Thuat toan | Tim thay duong | Do dai duong di | Node da duyet | Thoi gian |
| --- | --- | ---: | ---: | ---: |
| BFS | ... | ... | ... | ... |
| A* + Manhattan | ... | ... | ... | ... |
| A* + ANN | ... | ... | ... | ... |

## 12. Nhan xet va han che

- BFS dam bao tim duong ngan nhat nhung duyet nhieu node.
- A* + Manhattan thuong duyet it node hon BFS.
- A* + ANN co the giam node duyet trong mot so truong hop.
- ANN heuristic khong dam bao luon toi uu vi co the du doan sai.

## 13. Ket luan

- Project the hien duoc cach ANN hoc tu du lieu duoc gan nhan tu thuat toan truyen thong.
- ANN duoc ung dung lam heuristic cho A*.
- Ket qua cho thay su giao thoa giua AI hien dai va thuat toan tim kiem co dien.

## 14. Muc do su dung AI ho tro

- AI ho tro tao khung project, goi y cau truc code va cach trinh bay.
- Nhom can tu chay code, doc ket qua, dien bang so lieu va giai thich ket qua thuc nghiem.
- Khong copy nguyen van dinh nghia ly thuyet tu AI vao slide/bao cao.
