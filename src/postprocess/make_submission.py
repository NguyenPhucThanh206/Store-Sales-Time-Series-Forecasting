"""Hậu xử lý và tạo file nộp Kaggle.

Chạy từ thư mục gốc repo:
    # Tuần 1: nộp riêng LightGBM
    python -m src.postprocess.make_submission --pred predictions/test/test_pred_lgbm.npy \
        --out submissions/submission_w1.csv

    # Bản cuối: kết quả Blending (đường dẫn do bạn Trí chốt)
    python -m src.postprocess.make_submission --pred predictions/test/test_pred_blend.npy \
        --out submissions/submission_final.csv

Các bước:
  1) Đọc dự đoán thang log1p, đổi ngược về doanh số thật bằng expm1, cắt về >= 0.
  2) Ép dự đoán = 0 cho chuỗi toàn 0 (và chuỗi 0 liên tục ở đuôi) trong zero_series.csv.
  3) Ghép với id, sắp theo thứ tự của sample_submission.csv, kiểm tra định dạng rồi ghi CSV.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT / "data" / "processed"
RAW_DIR = ROOT / "data" / "raw"


def build_submission(pred_log, test_clean, zero_info, sample, zero_tail=True) -> pd.DataFrame:
    pred_log = np.asarray(pred_log, dtype="float64")
    if pred_log.shape != (len(test_clean),):
        raise ValueError(f"Dự đoán có shape {pred_log.shape}, test_clean có {len(test_clean)} dòng. "
                         "Thứ tự phải theo test_clean (sắp theo series_id, date).")
    if not np.isfinite(pred_log).all():
        raise ValueError("Dự đoán có NaN hoặc inf")

    sub = pd.DataFrame({"id": test_clean["id"].to_numpy(),
                        "sales": np.clip(np.expm1(pred_log), 0, None)})

    kinds = ["all_zero"] + ([k for k in zero_info["kind"].unique() if k.startswith("zero_last")]
                            if zero_tail else [])
    zero_ids = set(zero_info.loc[zero_info["kind"].isin(kinds), "series_id"])
    mask = test_clean["series_id"].isin(zero_ids).to_numpy()
    sub.loc[mask, "sales"] = 0.0
    print(f"Ép về 0: {len(zero_ids)} chuỗi ({int(mask.sum())} dòng); kiểu: {kinds}")

    sub = sub.set_index("id").loc[sample["id"]].reset_index()   # đúng thứ tự sample_submission
    assert list(sub.columns) == list(sample.columns) == ["id", "sales"]
    assert len(sub) == len(sample) and sub["id"].is_unique
    assert set(sub["id"]) == set(sample["id"])
    assert sub["sales"].notna().all() and (sub["sales"] >= 0).all()
    return sub


def run(pred_path, out_path, proc_dir=PROC_DIR, raw_dir=RAW_DIR, zero_tail=True):
    proc_dir, raw_dir = Path(proc_dir), Path(raw_dir)
    test_clean = pd.read_parquet(proc_dir / "test_clean.parquet")
    zero_info = pd.read_csv(proc_dir / "zero_series.csv")
    sample = pd.read_csv(raw_dir / "sample_submission.csv")
    pred = np.load(pred_path)

    sub = build_submission(pred, test_clean, zero_info, sample, zero_tail)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sub.to_csv(out_path, index=False)
    print(f"Đã ghi {out_path} {sub.shape} | sales: min {sub['sales'].min():.2f}, "
          f"trung bình {sub['sales'].mean():.2f}, max {sub['sales'].max():.2f}")
    return sub


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True, help="File .npy dự đoán test (thang log1p)")
    p.add_argument("--out", required=True, help="File CSV đầu ra, vd submissions/submission_w1.csv")
    p.add_argument("--no-zero-tail", action="store_true",
                   help="Chỉ ép 0 cho chuỗi toàn 0, không ép chuỗi 0 ở 28 ngày cuối")
    a = p.parse_args()
    run(a.pred, a.out, zero_tail=not a.no_zero_tail)
