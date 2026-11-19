**ĐẠI HỌC QUỐC GIA HÀ NỘI**  
**TRƯỜNG ĐẠI HỌC CÔNG NGHỆ**

---

# KẾ HOẠCH TRIỂN KHAI VÀ PHÂN CÔNG NHIỆM VỤ DỰ ÁN MÔN HỌC

**Môn học:** Nhập môn Học máy (INT3405)  
**Đề tài:** Dự báo Doanh số Bán lẻ Chuỗi thời gian (Store Sales – Time Series Forecasting)

| Hạng mục | Chi tiết |
| :--- | :--- |
| **Tên nhóm** | Nhóm 4 (`UET_[INT3405E_5]_[STT]`) |
| **Thành viên** | Nguyễn Phúc Thành (24021629)<br>An Đức Minh Trí (24021645)<br>Nguyễn Mạnh Kiên (24021541)<br>Doãn Duy Lợi (24021549) |
| **Nhóm trưởng** | Nguyễn Phúc Thành |

**HÀ NỘI - 11/2026**

---

## MỤC LỤC

* **CHƯƠNG 1. TỔNG QUAN VỀ BÀI TOÁN VÀ MỤC TIÊU DỰ ÁN**
  * 1.1. Giới thiệu chung về bài toán
  * 1.2. Dữ liệu đầu vào và Thước đo đánh giá (RMSLE)
  * 1.3. Quy trình triển khai và Tỷ trọng tiến độ 3 tuần
* **CHƯƠNG 2. GIẢI PHÁP KỸ THUẬT VÀ KIẾN TRÚC MÔ HÌNH**
  * 2.1. Pipeline Tiền xử lý và Kỹ thuật Đặc trưng (Feature Engineering)
  * 2.2. Kiến trúc 2 nhánh mô hình (3 Gradient Boosting & GRU)
  * 2.3. Chiến lược Kết hợp (Ensembling) và Giải thích Mô hình (XAI)
* **CHƯƠNG 3. KẾ HOẠCH TRIỂN KHAI VÀ PHÂN CÔNG NHIỆM VỤ**
  * 3.1. Phân công vai trò và Quyền sở hữu mã nguồn
  * 3.2. Bảng hạng mục công việc (Action Plan)
  * 3.3. Ước tính nỗ lực theo từng thành viên
  * 3.4. Phụ thuộc, lịch bàn giao và quy tắc phối hợp
  * 3.5. Các mốc kiểm tra, tiêu chí nghiệm thu và phương án dự phòng

---

## CHƯƠNG 1. TỔNG QUAN VỀ BÀI TOÁN VÀ MỤC TIÊU DỰ ÁN

### 1.1. Giới thiệu chung về bài toán
Dự án giải quyết bài toán dự báo chuỗi thời gian đa biến từ cuộc thi **Store Sales – Time Series Forecasting** trên Kaggle. Mục tiêu là dự báo doanh số bán hàng (`sales`) trong **16 ngày tương lai** cho **1.782 chuỗi thời gian** độc lập, tương ứng với tổ hợp của **54 cửa hàng** (`store_nbr`) và **33 nhóm mặt hàng** (`family`) thuộc hệ thống siêu thị Corporación Favorita tại Ecuador.

### 1.2. Dữ liệu đầu vào và Thước đo đánh giá
Hệ thống sử dụng 6 tệp dữ liệu chuẩn (`train.csv`, `test.csv`, `stores.csv`, `oil.csv`, `holidays_events.csv`, `sample_submission.csv`). Chất lượng mô hình được đánh giá bằng chỉ số **RMSLE** (Root Mean Squared Logarithmic Error):

$$\text{RMSLE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} \left( \log(1 + \hat{y}_i) - \log(1 + y_i) \right)^2}$$

Để tối ưu trực tiếp thước đo này, biến mục tiêu được biến đổi sang không gian logarit $z_i = \log(1 + y_i)$ trước khi huấn luyện và chuyển ngược lại bằng $\hat{y}_i = \max(0, \exp(\hat{z}_i) - 1)$ khi xuất kết quả.

### 1.3. Quy trình triển khai và Tỷ trọng tiến độ 3 tuần
Kế hoạch thực hiện kéo dài trong **3 tuần (21 ngày)** với phân bổ tỷ trọng rõ ràng:
* **Tuần 1 (Chiếm 80% khối lượng lập trình):** Hoàn thiện khung tiền xử lý dữ liệu, tạo đặc trưng dùng chung, huấn luyện hoàn chỉnh cả 4 mô hình (LightGBM, CatBoost, XGBoost, GRU) và **bắt buộc xuất tệp `submission.csv` nộp lên Kaggle** để lấy điểm mốc trên Public Leaderboard.
* **Tuần 2 (Chiếm 20% khối lượng lập trình):** Tinh chỉnh siêu tham số, chạy Ensembling (trộn dự đoán giữa 3 mô hình Gradient Boosting và GRU), triển khai XAI (SHAP) để giải thích mô hình và đóng băng mã nguồn.
* **Tuần 3 (Báo cáo & Slide):** Tổng hợp bảng số liệu thực nghiệm, vẽ sơ đồ kiến trúc, hoàn thiện slide thuyết trình và tổng duyệt phản biện.

---

## CHƯƠNG 2. GIẢI PHÁP KỸ THUẬT VÀ KIẾN TRÚC MÔ HÌNH

Hệ thống được thiết kế theo mô hình **2 nhánh song song** trên cùng một bộ quy tắc chia tập Validation (sử dụng **16 ngày cuối cùng** của tập `train.csv` làm tập kiểm định cục bộ để khớp với độ dài 16 ngày của tập `test.csv`).

### 2.1. Pipeline Tiền xử lý và Kỹ thuật Đặc trưng (Feature Engineering)
* **Làm sạch dữ liệu:** Cắt bỏ phần chuỗi toàn số `0` trước ngày khai trương thực tế của từng cặp `(store_nbr, family)`; nội suy tuyến tính giá dầu thô (`oil.csv`) cho các ngày nghỉ cuối tuần; loại bỏ các ngày lễ có trạng thái `transferred == True`.
* **Khớp phạm vi địa lý ngày lễ:** Phân tách chính xác ngày lễ `National` (toàn quốc), `Regional` (theo `state`), và `Local` (theo `city`) từ `holidays_events.csv` kết hợp `stores.csv`.
* **Đặc trưng miền nghiệp vụ & Thời gian:** Biến nhận lương `is_payday` (ngày 15 và ngày cuối tháng), `dayofweek`, `days_to_holiday`, trung bình trượt giá dầu (`oil_ma7`, `oil_ma14`), đặc trưng khuyến mãi (`onpromotion`, `onpromotion_ma7`, `onpromotion_lead1`) và các đặc trưng trễ an toàn từ mốc $t - 16$ trở về trước (`sales_lag_16`, `sales_lag_21`, `sales_lag_28`, `sales_lag_364`, `rolling_mean_7`, `rolling_mean_28`).

### 2.2. Kiến trúc 2 nhánh mô hình
1. **Nhánh 1 (Non-DL – Bộ ba Gradient Boosting):**
   * **LightGBM:** Tận dụng tốc độ huấn luyện nhanh theo chiến lược *leaf-wise* để làm mô hình cơ sở (Baseline) và đánh giá độ hiệu quả của từng nhóm đặc trưng.
   * **CatBoost:** Tận dụng cơ chế cây đối xứng (*oblivious trees*) và mã hóa trực tiếp các biến phân loại định danh (`store_nbr`, `family`, `city`, `cluster`) qua tham số `cat_features`.
   * **XGBoost:** Sử dụng thuật toán xấp xỉ histogram (`tree_method='hist'`, `enable_categorical=True`) theo chiến lược *depth-wise* để tạo sự đa dạng cấu trúc cây khi kết hợp.
2. **Nhánh 2 (DL-based – Global Multi-Horizon GRU):**
   * Mô hình mạng nơ-ron hồi quy **GRU** học toàn cục trên toàn bộ 1.782 chuỗi.
   * Sử dụng các lớp `Embedding` để biểu diễn thông tin tĩnh (`store_nbr`, `family`, `cluster`), kết hợp với chuỗi đặc trưng động trong cửa sổ trượt $L = 30$ ngày quá khứ và xuất trực tiếp vector dự báo **16 ngày** qua lớp Linear cuối cùng (tránh sai số tích lũy).

### 2.3. Chiến lược Kết hợp (Ensembling) và Giải thích Mô hình (XAI)
* **Blending 2 tầng:** Tìm trọng số tối ưu trên tập Validation để trộn 3 mô hình nhánh Non-DL ($\hat{z}_{\text{Non-DL}}$), sau đó kết hợp với nhánh Deep Learning ($\hat{z}_{\text{GRU}}$) trước khi biến đổi ngược $\exp(\hat{z}) - 1$. Áp dụng hậu xử lý ép doanh số ngày 1/1 về `0`.
* **XAI (Explainable AI):** Sử dụng `shap.TreeExplainer` trên mô hình tốt nhất của nhánh Non-DL để xuất biểu đồ **Summary Plot** (đánh giá mức độ đóng góp toàn cục của đặc trưng) và **Waterfall Plot** (giải thích cục bộ biến động doanh số trong dịp lễ/ngày lương).

---

## CHƯƠNG 3. KẾ HOẠCH TRIỂN KHAI VÀ PHÂN CÔNG NHIỆM VỤ

### 3.1. Phân công vai trò và Quyền sở hữu mã nguồn
Để 4 thành viên làm việc song song mà không xảy ra xung đột mã nguồn, mỗi thành viên sở hữu độc lập một phân hệ trong kho lưu trữ GitHub của nhóm (Bảng 3.1).

**Bảng 3.1. Phân công vai trò và thư mục phụ trách của các thành viên**

| Thành viên (MSSV) | Vai trò chính | Thư mục sở hữu | Phần kiến trúc phụ trách |
| :--- | :--- | :--- | :--- |
| **Nguyễn Phúc Thành**<br>*(24021629 - Nhóm trưởng)* | Điều phối tiến độ, Dựng khung dữ liệu, **LightGBM Lead**, Quản lý nộp bài Kaggle | `src/preprocess/`, `src/models/lgbm.py`, `submissions/` | Khung dữ liệu chuẩn, chia tập Validation, nhánh Non-DL (LightGBM), hậu xử lý & Submit |
| **An Đức Minh Trí**<br>*(24021645)* | **Feature Engineering Lead**, **CatBoost Lead**, Phối hợp Ensembling | `src/features/`, `src/models/catboost.py`, `src/ensemble/` | Xử lý ngày lễ/giá dầu, tạo bộ đặc trưng chung, nhánh Non-DL (CatBoost), xây dựng khung code Ensembling |
| **Nguyễn Mạnh Kiên**<br>*(24021541)* | **XGBoost Lead**, **XAI (SHAP) Lead**, **Tối ưu Ensembling**, Phân tích lỗi | `src/models/xgboost.py`, `src/xai/`, `src/ensemble/` | Nhánh Non-DL (XGBoost), tối ưu tham số cây, triển khai SHAP, phân tích ma trận tương quan & tối ưu trọng số Blending |
| **Doãn Duy Lợi**<br>*(24021549)* | **Deep Learning (GRU) Lead** | `src/dl_gru/` | Pipeline cửa sổ trượt 30 ngày + Embedding, kiến trúc và huấn luyện mạng GRU |

---

### 3.2. Bảng hạng mục công việc (Action Plan)
Quy ước: **1 ngày công $\approx$ 3–4 giờ làm việc tập trung**. Tuần 1 (Ngày 1–7) hoàn thành 80% khối lượng code; Tuần 2 (Ngày 8–14) hoàn thành 20% tối ưu & XAI; Tuần 3 (Ngày 15–21) hoàn thiện báo cáo & slide.

**Bảng 3.2. Danh sách hạng mục công việc, nỗ lực và phân công thực hiện**

| Mã | Hạng mục công việc | Nội dung kỹ thuật chính | Nỗ lực (ngày công) | Phụ trách | Thời gian | Mốc tiến độ |
| :--- | :--- | :--- | :---: | :--- | :---: | :---: |
| **T01** | Khởi tạo nhóm Kaggle & Repo GitHub | Lập team `UET_[INT3405E_5]_[STT]`, cấu trúc thư mục, quy ước nhánh code | 1,0 | **Thành** | Tuần 1 (N1) | Tuần 1 (80%) |
| **T02** | Khung tiền xử lý & Chia tập Validation | Đọc 6 file CSV, biến đổi `log1p(sales)`, cắt chuỗi 0 đầu kỳ, tách 16 ngày cuối làm Validation | 2,0 | **Thành** | Tuần 1 (N1–N2) | Tuần 1 (80%) |
| **T03** | EDA & Xử lý dữ liệu ngoại sinh | Nội suy `oil.csv`, lọc `Transferred` và khớp địa lý `holidays_events.csv` | 2,0 | **Trí** | Tuần 1 (N1–N2) | Tuần 1 (80%) |
| **T04** | Pipeline Dataset & Embedding cho GRU | Xây dựng bộ dữ liệu cửa sổ trượt 30 ngày $\rightarrow$ 16 ngày, tầng Embedding cho `store_nbr`, `family` | 2,5 | **Lợi** | Tuần 1 (N1–N3) | Tuần 1 (80%) |
| **T05** | Bộ Feature Engineering dùng chung | Tạo `is_payday`, `days_to_holiday`, `lag_16/21/28/364`, `rolling_mean`, xuất file `.parquet` | 3,0 | **Trí** | Tuần 1 (N3–N4) | Tuần 1 (80%) |
| **T06** | Huấn luyện mô hình LightGBM (Baseline & Full) | Chạy LightGBM trên dữ liệu sạch, đo RMSLE Validation, lưu mảng dự báo `.npy` | 2,5 | **Thành** | Tuần 1 (N3–N5) | Tuần 1 (80%) |
| **T07** | Huấn luyện mô hình XGBoost | Cấu hình `hist` & `enable_categorical`, huấn luyện trên tập `.parquet`, lưu dự báo `.npy` | 3,0 | **Kiên** | Tuần 1 (N4–N6) | Tuần 1 (80%) |
| **T08** | Huấn luyện mô hình CatBoost | Cấu hình `cat_features`, huấn luyện trên tập `.parquet`, lưu dự báo `.npy` | 2,5 | **Trí** | Tuần 1 (N5–N6) | Tuần 1 (80%) |
| **T09** | Dựng & Huấn luyện mạng GRU hoàn chỉnh | Huấn luyện mô hình Global GRU trên GPU (Kaggle/Colab), xuất dự báo Validation & Test `.npy` | 3,5 | **Lợi** | Tuần 1 (N4–N7) | Tuần 1 (80%) |
| **T10** | **Xuất file `submission.csv` nộp Kaggle lần 1** | Hậu xử lý (`expm1`, ép ngày 1/1 về 0), nộp kết quả lên Kaggle lấy điểm mốc Public LB | 1,0 | **Thành** | Tuần 1 (N6–N7) | **Chốt mốc 80%** |
| **T11** | Tinh chỉnh siêu tham số nhánh Non-DL | Chọn lọc đặc trưng quan trọng, tinh chỉnh `learning_rate`, `num_leaves`, `max_depth` | 1,5 | **Thành, Trí, Kiên** | Tuần 2 (N8–N10) | Tuần 2 (20%) |
| **T12** | Tinh chỉnh nhẹ kiến trúc GRU | Thử nghiệm Dropout, kích thước lớp ẩn hoặc lịch giảm learning rate để giảm lỗi Validation | 1,5 | **Lợi** | Tuần 2 (N8–N10) | Tuần 2 (20%) |
| **T13** | Chạy Ensembling (Blending 2 nhánh) | Phân tích tương quan lỗi, tìm trọng số tối ưu kết hợp `LGBM + CatBoost + XGBoost` và `GRU` | 1,5 | **Kiên, Trí** | Tuần 2 (N11–N12) | Tuần 2 (20%) |
| **T14** | Triển khai XAI (SHAP) & Phân tích lỗi | Chạy `TreeExplainer`, xuất Summary Plot, Waterfall Plot, phân tích sai số dư theo nhóm | 2,0 | **Kiên** | Tuần 2 (N10–N13) | Tuần 2 (20%) |
| **T15** | Nộp bản cuối Kaggle & Đóng băng mã nguồn | Nộp file `submission.csv` đạt điểm cao nhất, kiểm tra tính tái lập của toàn bộ notebook | 1,0 | **Thành** | Tuần 2 (N13–N14) | **Chốt mốc 100% code** |
| **T16** | Soạn thảo nội dung Slide & Báo cáo | Viết nội dung: Tổng quan/Kết quả (Thành: 0.5), FE/CatBoost (Trí: 0.3), GRU (Lợi: 1.0), XGBoost/XAI/Blending (Kiên: 1.2) | 3,0 | **Cả nhóm** | Tuần 3 (N15–N18) | Tuần 3 (Báo cáo) |
| **T17** | Thiết kế đồ họa Slide & Tổng duyệt thuyết trình | Căn chỉnh layout/đồ họa XAI (Kiên: 0.8), vẽ sơ đồ DL (Lợi: 0.5), tổng duyệt (Thành: 0.5, Trí: 0.2) | 2,0 | **Cả nhóm** | Tuần 3 (N19–N21) | **Hoàn tất dự án** |
| | **Tổng cộng nỗ lực toàn dự án** | | **35,5** | | | |

---

### 3.3. Ước tính nỗ lực theo từng thành viên

**Bảng 3.3. Tổng hợp nỗ lực phân bổ theo từng thành viên (đơn vị: ngày công)**

| Thành viên | Hạng mục chuyên môn riêng | Hạng mục phối hợp (T11, T16, T17) | Tổng nỗ lực | Tỷ trọng đóng góp |
| :--- | :---: | :---: | :---: | :---: |
| **Nguyễn Phúc Thành** | 6,5 *(T01, T02, T06, T10, T15)* | 2,0 *(T11: 0.5, T16: 0.5, T17: 0.5 + Điều phối: 0.5)* | **8,5** | 23,9% |
| **An Đức Minh Trí** | 8,0 *(T03: 2.0, T05: 3.0, T08: 2.5, T13: 0.5)* | 1,0 *(T11: 0.5, T16: 0.3, T17: 0.2)* | **9,0** | 25,4% |
| **Nguyễn Mạnh Kiên** | 6,0 *(T07: 3.0, T13: 1.0, T14: 2.0)* | 3,0 *(T11: 0.5, T16: 1.2, T17: 0.8)* | **9,0** | 25,4% |
| **Doãn Duy Lợi** | 7,5 *(T04: 2.5, T09: 3.5, T12: 1.5)* | 1,5 *(T16: 1.0, T17: 0.5)* | **9,0** | 25,4% |
| **Tổng cộng** | **28,0** | **7,5** | **35,5** | **100%** |

---

### 3.4. Phụ thuộc, lịch bàn giao và quy tắc phối hợp
Để tránh tắc nghẽn giữa khâu xử lý dữ liệu và huấn luyện mô hình, nhóm thống nhất lịch bàn giao các tệp trung gian theo **Bảng 3.4**.

**Bảng 3.4. Lịch bàn giao sản phẩm trung gian giữa các thành viên**

| Thời điểm | Nội dung bàn giao | Người bàn giao $\rightarrow$ Người nhận | Định dạng tệp chuẩn |
| :---: | :--- | :--- | :--- |
| **Cuối Ngày 2** | Khung dữ liệu đã làm sạch, biến đổi `log1p` và danh sách ngày Validation chuẩn | **Thành** $\rightarrow$ **Cả nhóm** | `train_clean.parquet`, `val_dates.json` |
| **Cuối Ngày 4** | Bộ dữ liệu đầy đủ Feature Engineering dùng chung cho nhánh Non-DL | **Trí** $\rightarrow$ **Thành, Kiên** | `train_fe.parquet`, `test_fe.parquet` |
| **Cuối Ngày 6** | Mảng dự báo trên tập Validation và Test của cả 3 mô hình Gradient Boosting | **Thành, Trí, Kiên** $\rightarrow$ **Thành** | `val_pred_[model].npy`, `test_pred_[model].npy` |
| **Cuối Ngày 7** | Tệp dự báo Kaggle tuần 1 và mảng dự báo bản đầu của mô hình GRU | **Thành** (nộp Kaggle); **Lợi** $\rightarrow$ **Trí** | `submission_w1.csv`, `val_pred_gru.npy` |
| **Cuối Ngày 10** | Các mảng dự báo sau khi tinh chỉnh (Tuần 2) để chạy Ensembling tổng hợp | **Thành, Kiên, Lợi** $\rightarrow$ **Trí** | `val_pred_[model]_v2.npy`, `test_pred_[model]_v2.npy` |
| **Cuối Ngày 13** | Tệp kết quả trộn cuối cùng (Final Ensemble) và bộ biểu đồ SHAP chất lượng cao | **Trí, Kiên** $\rightarrow$ **Thành** | `submission_final.csv`, `shap_*.png` |
| **Cuối Ngày 18** | Bản nháp nội dung và biểu đồ từng phần để ghép vào Slide chung | **Trí, Kiên, Lợi** $\rightarrow$ **Thành** | File `.pptx` / Google Slides |

**Quy tắc phối hợp kỹ thuật:**
1. **Thống nhất không gian dự báo:** Mọi tệp `.npy` lưu trữ dự báo nội bộ bắt buộc nằm trên thang $\log(1 + \text{sales})$ và giữ đúng thứ tự dòng của tập Validation và `test.csv`.
2. **Quản lý mã nguồn:** Không đẩy trực tiếp các tệp dữ liệu lớn (`.csv`, `.parquet`, `.npy`) lên GitHub (đưa vào `.gitignore`), chỉ lưu trữ trên Google Drive chung của nhóm; GitHub chỉ chứa mã nguồn `.py`/`.ipynb` và tệp `requirements.txt`.

---

### 3.5. Các mốc kiểm tra, tiêu chí nghiệm thu và phương án dự phòng

**Bảng 3.5. Các mốc kiểm tra tiến độ (Milestones)**

| Mốc | Thời điểm | Kết quả bắt buộc phải đạt | Tiêu chí nghiệm thu |
| :---: | :--- | :--- | :--- |
| **M1** | **Cuối Tuần 1 (Ngày 7)** | Hoàn thành **80% khối lượng code**; chạy xong cả 4 mô hình; **đã nộp `submission.csv` lên Kaggle** | Có điểm Public Leaderboard hợp lệ trên Kaggle; cả 4 mô hình đều có điểm RMSLE cục bộ trên tập Validation |
| **M2** | **Cuối Tuần 2 (Ngày 14)** | Hoàn thành **100% khối lượng code**; chốt điểm Final Ensemble và bộ biểu đồ XAI | Điểm RMSLE của tổ hợp (Ensemble) thấp hơn từng mô hình đơn lẻ; có đầy đủ biểu đồ SHAP (Summary & Waterfall) |
| **M3** | **Cuối Tuần 3 (Ngày 21)** | Hoàn thiện Slide báo cáo và tổng duyệt thuyết trình | Slide đầy đủ các mục theo yêu cầu môn học INT3405, đảm bảo thời lượng trình bày và trả lời trôi chảy các câu hỏi kỹ thuật |

**Phương án dự phòng rủi ro:**
* **Rủi ro trễ dữ liệu Feature Engineering (Ngày 4):** Nếu bộ đặc trưng nâng cao của Trí chưa hoàn thiện kịp Ngày 4, Thành và Kiên vẫn tiến hành huấn luyện LightGBM và XGBoost trên bộ dữ liệu cơ bản từ Ngày 2 (`train_clean.parquet`) để đảm bảo chắc chắn có tệp `submission.csv` nộp lên Kaggle đúng hạn Ngày 7, sau đó cập nhật đặc trưng mới ở đầu Tuần 2.
* **Rủi ro thiếu tài nguyên GPU khi huấn luyện GRU:** Lợi chủ động sử dụng song song tài nguyên GPU miễn phí của Kaggle (30 giờ/tuần) và Google Colab; trong trường hợp mạng GRU hội tụ chậm, ưu tiên sử dụng kiến trúc GRU 1–2 lớp với kích thước ẩn vừa phải (64–128) và dùng `EarlyStopping` để đảm bảo hoàn thành đúng mốc Ngày 7.