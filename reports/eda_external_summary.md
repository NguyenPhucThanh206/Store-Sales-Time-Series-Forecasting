# T03 - EDA dữ liệu ngoại sinh

Sinh tự động bởi `python -m src.features.eda_external`.

- Giá dầu: 529 / 1704 ngày thiếu giá được nội suy tuyến tính (chủ yếu cuối tuần).
- Tương quan giữa tổng doanh số (TB 28 ngày) và giá dầu: **-0.81**.

## Tác động ngày lễ (so với trung bình tháng của cùng cửa hàng)

| nhóm | số (store, ngày) | chênh lệch % so với TB tháng | chênh lệch % (đã khử thứ trong tuần) |
| --- | --- | --- | --- |
| Ngày thường | 76618 | -0.7 | +0.0 |
| Lễ National | 3699 | +14.1 | +14.5 |
| Lễ Regional | 28 | +11.2 | +9.9 |
| Lễ Local | 332 | -5.2 | -5.0 |
| Bridge | 148 | +14.0 | +25.4 |
| Additional | 1562 | +21.1 | +19.2 |
| Event | 2803 | +3.4 | +1.0 |
| Work Day | 240 | +20.0 | -2.1 |

## Ngày trong tuần (ngày thường, %)

T2: -1.6 | T3: -8.2 | T4: -8.0 | T5: -17.7 | T6: -9.4 | T7: +22.4 | CN: +26.3

## Ngày lương: +1.2% (ngày khác: -0.8%)

Biểu đồ: `reports/figures/eda_oil_interpolation.png`, `eda_sales_vs_oil.png`, `eda_holiday_effect.png`, `eda_dayofweek_payday.png`.
