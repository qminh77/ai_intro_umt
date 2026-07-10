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

Neu terminal khong nhan lenh `python`, dung truc tiep interpreter trong moi truong ao:

```bash
./.venv/bin/python -m src.dataset --help
```

## Lam Sach Ket Qua Cu

Da co the xoa sach dataset/model/output va chay lai tu dau. Lenh tuong duong:

```bash
rm -f data/*.csv
rm -f models/*.keras models/*.h5 models/*.tflite
rm -f outputs/*.csv outputs/*.json outputs/*.png outputs/*.txt
```

Sau khi clean, cac file sinh lai se nam trong `data/`, `models/`, `outputs/` nhung khong bi git track.

## Cach Chay

Sinh dataset me cung, trong do nhan `true_distance` duoc tinh bang BFS tu goal:

```bash
python -m src.dataset --mazes 500 --max-samples-per-maze 200
```

Lenh dataset day du tham so:

```bash
python -m src.dataset \
  --mazes 500 \
  --height 20 \
  --width 20 \
  --wall-prob 0.25 \
  --min-goal-distance 20 \
  --max-samples-per-maze 200 \
  --seed 42 \
  --output data/maze_dataset.csv
```

Lenh test nhanh dataset nho:

```bash
python -m src.dataset \
  --mazes 20 \
  --height 10 \
  --width 10 \
  --wall-prob 0.2 \
  --min-goal-distance 6 \
  --max-samples-per-maze 50 \
  --output data/maze_dataset.csv
```

Train 3 mo hinh de minh hoa underfitting, overfitting va good fit:

```bash
python -m src.train --experiment all
```

Train tung mo hinh rieng:

```bash
python -m src.train --experiment underfit
python -m src.train --experiment overfit
python -m src.train --experiment goodfit
```

Ghi de so epoch tu dong lenh, huu ich khi test nhanh:

```bash
python -m src.train --experiment underfit --epochs 2
python -m src.train --experiment overfit --epochs 50
python -m src.train --experiment goodfit --epochs 20
```

Lenh train day du tham so:

```bash
python -m src.train \
  --dataset data/maze_dataset.csv \
  --experiment all \
  --seed 42 \
  --models-dir models \
  --outputs-dir outputs
```

Chay cross-validation cho mo hinh goodfit:

```bash
python -m src.train --cross-val --folds 5 --epochs 10
```

Lenh cross-validation day du tham so:

```bash
python -m src.train \
  --dataset data/maze_dataset.csv \
  --cross-val \
  --folds 5 \
  --epochs 10 \
  --batch-size 64 \
  --cv-output outputs/cross_validation.json
```

Chay demo so sanh BFS, A* Manhattan va A* ANN:

```bash
python -m src.demo
```

Lenh demo day du tham so:

```bash
python -m src.demo \
  --height 20 \
  --width 20 \
  --wall-prob 0.25 \
  --min-goal-distance 20 \
  --seed 123 \
  --model models/ann_heuristic_goodfit.keras \
  --output outputs/demo_path.png \
  --metrics outputs/demo_metrics.json
```

Chay UI Pygame tuy chon:

```bash
python -m src.ui
```

Lenh UI day du tham so:

```bash
python -m src.ui \
  --height 20 \
  --width 20 \
  --wall-prob 0.25 \
  --min-goal-distance 20 \
  --model models/ann_heuristic_goodfit.keras
```

## Quy Trinh Chay Lai Tu Dau

Chay day du de tao ket qua nop bai:

```bash
source .venv/bin/activate
rm -f data/*.csv
rm -f models/*.keras models/*.h5 models/*.tflite
rm -f outputs/*.csv outputs/*.json outputs/*.png outputs/*.txt
python -m src.dataset --mazes 500 --max-samples-per-maze 200
python -m src.train --experiment all
python -m src.train --cross-val --folds 5 --epochs 10
python -m src.demo
```

Chay nhanh de kiem tra code truoc:

```bash
source .venv/bin/activate
python -m src.dataset --mazes 20 --height 10 --width 10 --min-goal-distance 6 --max-samples-per-maze 50
python -m src.train --experiment underfit --epochs 1
python -m src.demo --height 10 --width 10 --min-goal-distance 6
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
