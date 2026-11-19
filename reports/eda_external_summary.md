# T03 - EDA dữ liệu ngoại sinh

Sinh tự động bởi `python -m src.features.eda_external`. Mức nền = trung bình log doanh số của cùng cửa hàng trong cùng tháng; "% đã khử thứ" trừ thêm hiệu ứng thứ trong tuần của từng cửa hàng (ước lượng trên ngày thường).

## Giá dầu

- 529 / 1704 ngày thiếu giá được nội suy tuyến tính (chủ yếu cuối tuần).
- Tương quan tổng doanh số (TB 28 ngày) với giá dầu: thô **-0.81**, sau khử xu hướng tuyến tính **-0.20**, theo % thay đổi hằng tháng **0.04**.
- Tương quan thô chủ yếu do hai chuỗi cùng có xu hướng theo thời gian (doanh số tăng, dầu giảm từ 2014-2015); sau khi khử xu hướng chỉ còn quan hệ yếu và theo biến động ngắn hạn thì gần như bằng 0. Không nên kỳ vọng nhiều vào `oil_ma*`.

## Tác động ngày lễ / sự kiện

| nhóm | số ngày | số (store, ngày) | % so với TB tháng | % đã khử thứ |
| --- | --- | --- | --- | --- |
| Ngày thường | 1547 | 76618 | -0.7 | +0.0 |
| Lễ National (nghỉ) | 46 | 2292 | +8.1 | +10.2 |
| Lễ Regional (nghỉ) | 18 | 28 | +11.2 | +9.9 |
| Lễ Local (nghỉ) | 89 | 233 | -1.9 | -3.3 |
| Bridge (mẫu nhỏ) | 3 | 148 | +14.0 | +25.4 |
| Additional (quanh Giáng sinh) | 38 | 1562 | +21.1 | +19.2 |
| Work Day | 5 | 240 | +20.0 | -2.1 |
| Event: động đất | 31 | 1643 | +5.6 | +4.8 |
| Event: mua sắm | 11 | 555 | -2.6 | -9.4 |
| Event: World Cup | 14 | 658 | +3.4 | -0.7 |

"(mẫu nhỏ)": ít hơn 5 ngày khác nhau, con số kém tin cậy. Work Day thường rơi vào thứ 7 nên chỉ cột "đã khử thứ" là có ý nghĩa.

## Thứ trong tuần (ngày thường, %)

T2: -1.6 | T3: -8.2 | T4: -8.0 | T5: -17.7 | T6: -9.4 | T7: +22.4 | CN: +26.3

## Ngày trong tháng / ngày lương (ngày thường, đã khử thứ, %)

- Theo ngày: ngày 1: +22.5, ngày 2: +17.9, ngày 3: +12.8, ngày 4: +9.7, ngày 5: +6.9, ngày 15: -1.0, ngày 16: +3.5, ngày 30: +2.0, ngày 31: +7.3
- Theo số ngày kể từ ngày lương: 0: +2.1, 1: +11.8, 2: +8.6, 3: +5.3, 4: +3.2, 5: +3.2, 6: +0.4
- Cờ `is_payday` (ngày 15 + cuối tháng): +2.1% (ngày khác: -0.1%).
- Đỉnh doanh số rơi vào **ngày 1-3 đầu tháng** (sau ngày lương cuối tháng) và giảm dần; bản thân ngày 15 / cuối tháng gần như không tăng. Gợi ý T05: dùng `day_of_month`, `days_since_payday` thay vì chỉ `is_payday`.

Biểu đồ: `reports/figures/eda_oil_interpolation.png`, `eda_sales_vs_oil.png`, `eda_holiday_effect.png`, `eda_dayofweek_payday.png`.
