# Store Sales – Time Series Forecasting (INT3405E_5 - Nhóm 4)

Dự án môn **Nhập môn Học máy (INT3405)** - Trường Đại học Công nghệ, ĐHQGHN (UET-VNU).
* **Tên đội Kaggle:** `UET_[INT3405E_5]_[STT]`
* **Mục tiêu:** Dự báo doanh số 16 ngày tiếp theo cho 1.782 chuỗi thời gian (`54 cửa hàng x 33 nhóm sản phẩm`) của chuỗi siêu thị Corporación Favorita (Ecuador), tối ưu chỉ số **RMSLE**.

## Thành viên & Phân công
* **Nguyễn Phúc Thành (24021629 - Nhóm trưởng):** Tiền xử lý dữ liệu (`src/preprocess/`), LightGBM (`src/models/lgbm.py`), Hậu xử lý & Nộp bài Kaggle (`submissions/`).
* **An Đức Minh Trí (24021645):** Kỹ thuật đặc trưng (`src/features/`), CatBoost (`src/models/catboost.py`), Ensembling/Blending (`src/ensemble/`).
* **Nguyễn Mạnh Kiên (24021541):** XGBoost (`src/models/xgboost.py`), Giải thích mô hình XAI - SHAP (`src/xai/`).
* **Doãn Duy Lợi (24021549):** Nhánh Deep Learning - Global Multi-Horizon GRU (`src/dl_gru/`).

## Cấu trúc thư mục
```text
store-sales-forecasting/
├── data/
│   ├── raw/                  # Chứa 6 file CSV gốc từ Kaggle (không push lên Git)
│   └── processed/            # Chứa train_clean.parquet, train_fe.parquet, test_fe.parquet
├── predictions/
│   ├── val/                  # Chứa val_pred_[lgbm|catboost|xgboost|gru].npy (thang log1p)
│   └── test/                 # Chứa test_pred_[lgbm|catboost|xgboost|gru].npy (thang log1p)
├── submissions/              # Chứa submission_w1.csv, submission_final.csv
├── notebooks/                # Chứa Jupyter Notebooks nháp / chạy trên Kaggle GPU
├── reports/
│   └── figures/              # Chứa biểu đồ EDA và biểu đồ SHAP (Summary, Waterfall)
├── src/
│   ├── preprocess/           # T02: Làm sạch chuỗi 0, biến đổi log1p, tách 16 ngày Val
│   ├── features/             # T03, T05: Nội suy giá dầu, khớp lễ, tạo lag & rolling features
│   ├── models/               # T06, T07, T08: LightGBM, CatBoost, XGBoost
│   ├── dl_gru/               # T04, T09: Sliding window 30->16 ngày & mạng PyTorch GRU
│   ├── ensemble/             # T13: Blending 2 tầng giữa 3 GBDT và GRU
│   └── xai/                  # T14: Giải thích mô hình bằng TreeSHAP
├── .gitignore
├── CONTRIBUTING.md           # Quy tắc chia nhánh, commit, push & bàn giao dữ liệu
├── requirements.txt
└── README.md
```

## Cài đặt môi trường
```bash
python -m venv venv
# Trên Windows (PowerShell):
.\venv\Scripts\activate
pip install -r requirements.txt
```
