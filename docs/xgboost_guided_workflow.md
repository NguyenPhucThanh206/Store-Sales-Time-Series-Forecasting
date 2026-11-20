# Quy trình XGBoost T07

T05 bổ sung các nhóm trong `FEATURE_GROUPS` và đưa toàn bộ vào `FEATURES`;
`BASE_FEATURES` giữ nguyên tập cũ. Đầu ra mở rộng được tạo ở thư mục riêng để
đối chiếu với các parquet đã dùng cho kết quả trước. Trên máy hiện tại, bộ
mở rộng cũng đã được sao chép vào `data/processed/train_fe.parquet` và
`test_fe.parquet` để khớp `FEATURES`; bản 38 đặc trưng trước đó được giữ với
hậu tố `_baseline.parquet`. Lệnh sau tái tạo bộ mở rộng trong thư mục riêng:

Chuẩn bị các CSV của cuộc thi trong `data/raw`, tạo môi trường và chạy T02, T03
từ thư mục gốc của dự án:

```powershell
python -m venv .venv_xgb_cuda
.venv_xgb_cuda\Scripts\python.exe -m pip install -r requirements.txt "xgboost==3.0.5"
.venv_xgb_cuda\Scripts\python.exe -m src.preprocess.run_preprocess
.venv_xgb_cuda\Scripts\python.exe -m src.features.external
```

Sau đó dựng feature và huấn luyện:

```powershell
$proc = 'data\processed\xgb_guided'
New-Item -ItemType Directory -Force $proc | Out-Null
foreach ($name in @('train_clean.parquet','test_clean.parquet','oil_clean.parquet','holidays_store.parquet','zero_series.csv')) {
    Copy-Item -LiteralPath (Join-Path 'data\processed' $name) -Destination (Join-Path $proc $name)
}
.venv_xgb_cuda\Scripts\python.exe -m src.features.build_features --proc-dir $proc
.venv_xgb_cuda\Scripts\python.exe -m src.models.xgboost_guided --device cuda --workers 2
```

Môi trường `.venv_xgb_cuda` đã dùng XGBoost 3.0.5 (CUDA 12.5)
trên máy có driver CUDA 12.7; XGBoost 3.4.1 trong `.venv` được build bằng CUDA
13.3 và đã tự chuyển sang CPU. Có thể dùng `--device cpu` với môi trường có
đủ phụ thuộc. Tham số mặc định thử 20 cấu hình; kết quả từng fold được lưu ở
`predictions/experiments/guided_results.json` và chạy lại sẽ tiếp tục từ đó.
`--workers 2` chạy hai cấu hình tune đồng thời; dùng `--workers 1` nếu thiếu RAM.
Không tái sử dụng checkpoint khi thay phiên bản dữ liệu hoặc sửa logic/giới hạn
huấn luyện: dùng một `--out-dir` mới để kết quả được tính lại.

- Chọn có/không `base_margin`, các quyết định cấu trúc và lần lượt từng nhóm
  đặc trưng theo trung bình B/C/D. Ở mỗi bước thử đúng một thay đổi trên cấu
  hình đã chọn; chỉ giữ nhóm nếu điểm trung bình giảm.
- Chọn một trong 20 cấu hình theo B/C, giảm `eta` xuống 0,025 và xác định số
  vòng trên B/C/D. Chọn số vòng refit bằng trung vị số vòng tốt nhất × 1,05.
- Dự báo fold A theo thứ tự `is_val`, refit trên toàn bộ train cho ba seed,
  rồi xuất `predictions/val/xgb_val.npy`, `predictions/test/xgb_test.npy` và
  `submissions/submission_xgb_guided.csv`. Dự báo nội bộ ở thang log1p.
- Các lead khuyến mãi được tính lại chỉ trong cửa sổ 16 ngày khi validation;
  cuối cửa sổ không dùng khuyến mãi của ngày sau cửa sổ.

Các fold đang dùng dữ liệu T02 đã cắt chuỗi đầu kỳ và loại chuỗi toàn 0 theo
toàn bộ train, và dùng các đặc trưng ngoại sinh theo giả định của T03/T05.
Chúng không phải backtest thực tế nghiêm ngặt. Fold A cũng đã được xem trong
các thử nghiệm trước, nên không phải một tập kiểm tra hoàn toàn độc lập.

Lượt chạy thực tế đã hoàn tất: các fold B/C/D của cấu hình cuối lần lượt đạt
0,393875 / 0,379105 / 0,444041; trung bình 0,405674. Số vòng refit là 1.214;
fold A đạt **0,394913** trên 27.664 dòng. Mô hình dùng `base_margin`, trọng số
giảm theo thời gian và các nhóm same_weekday, long_history, yearly,
cross_section, promotion; không chọn nhóm sparse và transactions.

Ensemble trước đó đạt 0,390241 trên fold A với cách tính feature cũ. Khi tính
lại dự báo validation của ensemble cũ và giới hạn `onpromotion_lead1` trong
cùng 16 ngày như quy trình mới, điểm là **0,388730**. Vì vậy kết quả của hướng
mới hiện kém hơn trên fold A; cần đo thêm các fold với cùng quy trình trước
khi kết luận mô hình nào tổng quát hóa tốt hơn. Theo phản hồi của nhóm,
submission guided hiện cho kết quả Kaggle tốt nhất; điểm cụ thể chưa được ghi lại.
