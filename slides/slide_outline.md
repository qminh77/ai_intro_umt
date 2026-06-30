# Khung slide thuyet trinh 15-20 phut

## Slide 1: Ten de tai

- Ung dung ANN lam heuristic cho A* trong bai toan tim duong me cung.
- Ten thanh vien nhom.

## Slide 2: Ly do chon de tai

- Me cung la bai toan truc quan.
- A* la thuat toan tim kiem truyen thong.
- ANN giup hoc heuristic tu du lieu that.

## Slide 3: Bai toan tong quan

- Me cung 20x20.
- Start, Goal, tuong va duong di.
- Muc tieu: tim duong hop ly tu Start den Goal.

## Slide 4: A* hoat dong nhu the nao

- `f(n) = g(n) + h(n)`.
- Vai tro cua heuristic.

## Slide 5: Manhattan heuristic

- Cong thuc Manhattan.
- Diem manh: don gian, nhanh, co tinh chat tot trong me cung 4 huong.
- Han che: khong nhin thay vat can phuc tap.

## Slide 6: Y tuong thay heuristic bang ANN

- ANN nhan state cua o hien tai.
- ANN du doan chi phi con lai den Goal.
- A* dung gia tri nay lam `h(n)`.

## Slide 7: ANN hoc nhu the nao

- Forward propagation.
- Loss.
- Backpropagation.
- Epoch.

## Slide 8: Train / Validation / Test

- Training: hoc trong so.
- Validation: theo doi va chinh cau hinh.
- Test: danh gia cuoi cung.
- Ti le 70/15/15.

## Slide 9: Input / Output cua ANN

Input:

```text
x, y, goal_x, goal_y, dx, dy, up_wall, down_wall, left_wall, right_wall
```

Output:

```text
true_distance
```

## Slide 10: Tao dataset bang BFS

- Sinh nhieu me cung ngau nhien.
- Chon goal.
- Dung BFS tinh khoang cach that tu moi o den goal.
- Luu thanh CSV.

## Slide 11: Underfitting

- Mo hinh qua nho.
- Train it epoch.
- Chen hinh `outputs/underfit_loss.png`.
- Nhan xet train/validation loss.

## Slide 12: Overfitting

- Mo hinh qua lon.
- Train lau tren tap train nho.
- Chen hinh `outputs/overfit_loss.png`.
- Nhan xet train loss va validation loss.

## Slide 13: Good Fit

- Mo hinh vua phai.
- Dropout va EarlyStopping.
- Chen hinh `outputs/goodfit_loss.png`.
- Trinh bay Test MAE: khoang 2.27 buoc trong lan chay hien tai.

## Slide 14: Demo so sanh BFS, A* Manhattan, A* ANN

- Chen `outputs/demo_path.png`.
- Bang so sanh: path length, explored nodes, time.
- Lan chay hien tai: BFS 104 nodes, A* Manhattan 46 nodes, A* ANN 33 nodes.

## Slide 15: Ket luan va han che

- ANN co the hoc heuristic tu du lieu.
- A* + ANN la su ket hop giua AI hien dai va thuat toan co dien.
- Han che: ANN khong dam bao luon toi uu.
- Neu co them thoi gian: train nhieu me cung hon, them dac trung cuc bo rong hon.
