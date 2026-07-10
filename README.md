# Maze ANN A*

Project minh hoa cach dung ANN de hoc heuristic cho thuat toan A* trong bai toan me cung. Source da duoc rut gon de phu hop muc tieu thuyet trinh: tao dataset, train ANN underfit/overfit/goodfit, va demo ANN thay the heuristic Manhattan.

## Cau Truc Source

```text
src/
├── config.py   # Duong dan, cot feature, cau hinh mac dinh
├── maze.py     # Me cung, BFS, A*, Manhattan, SearchResult
├── dataset.py  # Tao dataset, trich 10 feature, gan nhan bang BFS
├── train.py    # Keras model, 3 thi nghiem, ve loss, cross-validation
├── demo.py     # ANN heuristic, so sanh BFS/A* Manhattan/A* ANN, ve hinh
└── ui.py       # Demo Pygame tuy chon
```

`slides/` va `reports/` la tai lieu thuyet trinh/bao cao, khong anh huong luong code chinh.

## Cai Dat

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Cach Chay

Sinh dataset me cung, trong do nhan `true_distance` duoc tinh bang BFS tu goal:

```bash
python -m src.dataset --mazes 500 --max-samples-per-maze 200
```

Train 3 mo hinh de minh hoa underfitting, overfitting va good fit:

```bash
python -m src.train --experiment all
```

Chay cross-validation cho mo hinh goodfit:

```bash
python -m src.train --cross-val --folds 5 --epochs 10
```

Chay demo so sanh BFS, A* Manhattan va A* ANN:

```bash
python -m src.demo
```

Chay UI Pygame tuy chon:

```bash
python -m src.ui
```

## Thiet Ke ANN

Input cua ANN gom 10 feature:

```text
x, y, goal_x, goal_y, dx, dy, up_wall, down_wall, left_wall, right_wall
```

Output cua ANN la `true_distance`: khoang cach ngan nhat tu vi tri hien tai den goal. Nhan nay duoc tao bang BFS nen co the dung lam muc tieu hoc cho ANN.

## Ba Thi Nghiem Chinh

| Thi nghiem | Muc dich | Cach tao hien tuong |
|---|---|---|
| `underfit` | Mo hinh hoc chua du | Mang rat nho, it epoch |
| `overfit` | Mo hinh hoc vet | Mang lon, train lau tren tap train nho |
| `goodfit` | Mo hinh can bang | Mang vua phai, Dropout, EarlyStopping |

Ket qua duoc luu vao:

```text
models/   # File .keras
outputs/  # Loss chart, history CSV, metrics JSON, demo image
```

## Y Tuong Demo Nang Cao

A* truyen thong dung:

```text
f(n) = g(n) + h(n)
```

Trong demo nay:

```text
h(n) = ANN(position, goal, local walls)
```

Nghia la ANN thay con nguoi uoc luong khoang cach con lai. Sau do so sanh so node da duyet giua BFS, A* Manhattan va A* ANN.
