# Tom tat ket qua chay thu nghiem

Ngay tao ket qua: 2026-06-30

## Dataset

- File: `data/maze_dataset.csv`
- So dong: 100,000
- So maze: 500
- Kich thuoc maze: 20x20
- Moi maze lay toi da 200 state co nhan.
- Nhan `true_distance` duoc tao bang BFS tu goal.

## Ket qua train ANN

| Mo hinh | Train rows | Validation rows | Test rows | Epochs ran | Test MSE | Test MAE |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Underfit | 70,000 | 15,000 | 15,000 | 5 | 16.253 | 2.619 |
| Overfit | 3,000 | 15,000 | 15,000 | 150 | 22.276 | 2.747 |
| Good fit | 70,000 | 15,000 | 15,000 | 15 | 14.416 | 2.268 |

Nhan xet:

- Underfit co mang qua nho va it epoch, nen sai so con cao.
- Overfit train tren it du lieu voi mang lon, train loss giam nhung validation/test khong tot.
- Good fit co ket qua tot nhat trong 3 mo hinh, Test MAE khoang 2.27 buoc.

## Cross-validation

Chay 5-fold cross-validation voi 10 epoch/fold:

| Fold | Validation rows | MAE | MSE |
| --- | ---: | ---: | ---: |
| 1 | 20,000 | 2.245 | 15.241 |
| 2 | 20,000 | 2.292 | 15.139 |
| 3 | 20,000 | 2.208 | 14.751 |
| 4 | 20,000 | 2.206 | 14.018 |
| 5 | 20,000 | 2.465 | 14.373 |

- Mean MAE: 2.283
- Mean MSE: 14.704

## Demo BFS, A* Manhattan, A* ANN

- Start: `(17, 1)`
- Goal: `(7, 7)`

| Thuat toan | Tim thay duong | Do dai duong di | Node da duyet | Thoi gian |
| --- | --- | ---: | ---: | ---: |
| BFS | Co | 20 | 104 | 0.09 ms |
| A* + Manhattan | Co | 20 | 46 | 0.06 ms |
| A* + ANN | Co | 20 | 33 | 681.70 ms |

Nhan xet:

- Ca 3 thuat toan deu tim duoc duong dai 20 buoc.
- A* + Manhattan duyet it node hon BFS.
- A* + ANN duyet it node nhat trong demo nay.
- Thoi gian A* + ANN cao hon vi moi lan uoc luong heuristic phai goi model Keras. Khi bao ve, nen nhan manh so node duyet la tieu chi chinh de minh hoa chat luong heuristic; thoi gian Python demo phu thuoc overhead goi model.

## File ket qua da tao

- `outputs/underfit_loss.png`
- `outputs/overfit_loss.png`
- `outputs/goodfit_loss.png`
- `outputs/demo_path.png`
- `outputs/training_summary.json`
- `outputs/cross_validation.json`
- `outputs/demo_metrics.json`

