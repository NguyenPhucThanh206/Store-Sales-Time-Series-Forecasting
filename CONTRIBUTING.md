# QUY TẮC LÀM VIỆC NHÓM & QUẢN LÝ MÃ NGUỒN (GIT WORKFLOW)
**Dự án:** Store Sales – Time Series Forecasting (INT3405E_5 - Nhóm 4)

---

## 1. Quyền sở hữu thư mục (Tránh Merge Conflict)
Mỗi thành viên chỉ chỉnh sửa code trong thư mục được phân công tại **Bảng 3.1**:

| Thành viên | Nhánh làm việc chính | Thư mục được phép sửa |
| :--- | :--- | :--- |
| **Nguyễn Phúc Thành** *(Leader)* | `feat/thanh-preprocess-lgbm` | `src/preprocess/`, `src/models/lgbm.py`, `submissions/` |
| **An Đức Minh Trí** | `feat/tri-fe-catboost-ensemble` | `src/features/`, `src/models/catboost.py`, `src/ensemble/` |
| **Nguyễn Mạnh Kiên** | `feat/kien-xgboost-xai` | `src/models/xgboost.py`, `src/xai/` |
| **Doãn Duy Lợi** | `feat/loi-dl-gru` | `src/dl_gru/` |

---

## 2. Quy tắc Push Code & Nhánh (Branching Rules)
1. **Tuyệt đối không push thẳng lên nhánh `main`:** Nhánh `main` chỉ nhận code đã chạy thành công thông qua Pull Request (PR) do Nhóm trưởng (Thành) duyệt.
2. **Quy trình làm việc hàng ngày:**
   ```bash
   # Bước 1: Cập nhật code mới nhất từ main về máy trước khi code
   git checkout main
   git pull origin main

   # Bước 2: Chuyển sang nhánh cá nhân và gộp thay đổi mới từ main
   git checkout feat/<ten-nhanh-cua-ban>
   git merge main

   # Bước 3: Sau khi code xong, kiểm tra kỹ các file chuẩn bị commit
   git status
   git add src/<thu-muc-cua-ban>/
   git commit -m "[T0X] Mô tả ngắn gọn công việc hoàn thành"
   git push origin feat/<ten-nhanh-cua-ban>
   ```
3. **Quy ước đặt tên Commit Message:**
   Gắn mã công việc (`[T01]` đến `[T17]` trong kế hoạch) ở đầu câu commit:
   * `[T01] chore: init project structure, gitignore and requirements`
   * `[T02] feat(preprocess): clean zero sales and split 16-day validation`
   * `[T05] feat(features): add payday, holiday geo-matching and lag 16-364`
   * `[T06] feat(lgbm): train baseline LightGBM and export val/test npy`

---

## 3. Quy tắc Dữ liệu & Định dạng File Trung gian (Bắt buộc)
1. **Không push dữ liệu lên GitHub:** Toàn bộ các tệp `.csv`, `.parquet`, `.npy`, `.cbm`, `.pt` đã được chặn trong `.gitignore`. Các tệp bàn giao theo **Bảng 3.4** (`train_clean.parquet`, `train_fe.parquet`, `val_pred_*.npy`, `test_pred_*.npy`) được tải lên **Google Drive chung của nhóm** theo đúng cấu trúc thư mục tương ứng.
2. **Chuẩn hóa mảng dự báo `.npy`:**
   * Mọi mảng dự báo `val_pred_[model].npy` và `test_pred_[model].npy` bắt buộc lưu ở **thang logarit** $z = \log(1 + 	ext{sales})$ (`np.log1p`).
   * Phải sắp xếp đúng thứ tự dòng (`store_nbr`, `family`, `date` hoặc `id`) của `val_dates.json` và `test.csv` để khâu Blending (`src/ensemble/blend.py`) không bị lệch dòng.
