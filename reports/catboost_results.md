# Kết quả CatBoost (T08)

Sinh tự động bởi `python -m src.models.catboost`.

| Chỉ số | Giá trị |
| --- | --- |
| Validation RMSLE (27.664 dòng, 2017-07-31 → 08-15) | **0.39772** |
| RMSLE tính cả 53 chuỗi toàn 0 (so với Kaggle) | 0.39177 |
| Số cây tốt nhất (early stopping) / huấn luyện lại | 1869 / 1962 |
| Đặc trưng | 38 (`base`) |
| Dữ liệu huấn luyện | 965,000 dòng, từ 2016-01-02 |
| Thời gian (bước 1 / tổng, CPU) | 11.7 / 22.6 phút |
| Dự báo val / test (log1p, trung bình) | 3.7273 / 3.5721 |

Tham số: `loss_function=RMSE`, `eval_metric=RMSE`, `learning_rate=0.1`, `depth=8`, `l2_leaf_reg=3.0`, `border_count=254`, `one_hot_max_size=8`, `max_ctr_complexity=2`, `bootstrap_type=Bernoulli`, `subsample=0.8`, `random_seed=42`.

Early stopping dùng chính tập Validation nên RMSLE ở trên hơi lạc quan (giống T06/T07).

## Thử nghiệm chọn cấu hình mặc định (bước 1, không huấn luyện lại)

| Bộ đặc trưng | Dữ liệu từ | lr | Val RMSLE | Số cây tốt nhất | Phút |
| --- | --- | ---: | ---: | ---: | ---: |
| base (38) | 2016-01-01 | 0.1 | **0.39772** | 1869 | 11.3 |
| all (59) | 2016-01-01 | 0.1 | 0.40076 | 2010 | 13.7 |
| all (59) | 2015-01-01 | 0.1 | 0.39955 | 2279 | 24.1 |
| base (38) | 2015-01-01 | 0.1 | 0.40660 | 324 (dừng sớm, od_wait 150) | 3.9 |
| all (59) | 2013 (toàn bộ) | 0.2 | 0.41387 | 304 | 6.6 |
| all (59) | 2015-01-01 | 0.2 | 0.40352 | 400 (chạm trần) | 3.3 |

Chênh lệch giữa các cấu hình gần nhau (~0,002-0,003) xấp xỉ độ nhiễu theo seed; khác biệt rõ
là dùng toàn bộ lịch sử từ 2013 kém hơn. Nhóm guided của T07 không giúp CatBoost trong thử nghiệm
này -> mặc định `base`, đồng thời tạo khác biệt với XGBoost (dùng guided) khi blending ở T13.

## 15 đặc trưng quan trọng nhất (PredictionValuesChange)

| # | Đặc trưng | Độ quan trọng |
| ---: | --- | ---: |
| 1 | `rolling_mean_28` | 23.92 |
| 2 | `rolling_mean_7` | 21.42 |
| 3 | `sales_lag_28` | 11.13 |
| 4 | `sales_lag_364` | 6.67 |
| 5 | `sales_lag_21` | 4.18 |
| 6 | `onpromotion_ma7` | 4.17 |
| 7 | `onpromotion` | 3.34 |
| 8 | `rolling_std_28` | 3.14 |
| 9 | `dayofyear` | 3.09 |
| 10 | `family` | 2.98 |
| 11 | `dayofweek` | 2.84 |
| 12 | `sales_lag_16` | 2.67 |
| 13 | `cluster` | 1.56 |
| 14 | `state` | 0.90 |
| 15 | `store_type` | 0.88 |
