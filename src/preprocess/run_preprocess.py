"""Chạy toàn bộ T02.  Từ thư mục gốc repo:

    python -m src.preprocess.run_preprocess
    python -m src.preprocess.run_preprocess --raw-dir data/raw --out-dir data/processed

Đầu ra (data/processed/):
    train_clean.parquet  - dạng long, cột: series_id, date, store_nbr, family, onpromotion,
                           sales, sales_log1p, is_closed, is_val
    test_clean.parquet   - id, date, store_nbr, family, onpromotion, series_id
    zero_series.csv      - chuỗi toàn 0 / 0 ở đuôi, dùng cho hậu xử lý
"""
import argparse
from pathlib import Path

from .clean import clean_train
from .load import OUT_DIR, RAW_DIR, load_raw
from .split import add_log1p, mark_validation


def run(raw_dir=RAW_DIR, out_dir=OUT_DIR, val_days=16, expected_series=1782, zero_tail_days=28):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = load_raw(raw_dir)
    print({k: v.shape for k, v in tables.items()})

    train, test_clean, zero_info, series_map = clean_train(
        tables["train"], tables["test"], zero_tail_days)
    train = add_log1p(train)
    train = mark_validation(train, val_days)

    # ---- kiểm tra trước khi bàn giao ----
    n_all_zero = int((zero_info["kind"] == "all_zero").sum())
    assert len(series_map) == expected_series, f"Có {len(series_map)} chuỗi, kỳ vọng {expected_series}"
    assert train["series_id"].nunique() + n_all_zero == expected_series
    assert not train.duplicated(["series_id", "date"]).any(), "Trùng (series_id, date)"
    open_days = train["is_closed"] == 0
    assert train.loc[open_days, "sales_log1p"].notna().all(), "NaN ngoài ngày đóng cửa"
    assert train["onpromotion"].notna().all()
    assert test_clean["series_id"].notna().all()

    train.to_parquet(out_dir / "train_clean.parquet", index=False)
    test_clean.to_parquet(out_dir / "test_clean.parquet", index=False)
    zero_info.to_csv(out_dir / "zero_series.csv", index=False)

    val = train[train["is_val"] == 1]
    print(f"train_clean: {len(train):,} dòng, {train['series_id'].nunique()} chuỗi "
          f"(+{n_all_zero} chuỗi toàn 0 bị loại)")
    print(f"validation : {val['date'].min().date()} -> {val['date'].max().date()} ({len(val):,} dòng)")
    print(f"zero_series: {zero_info['kind'].value_counts().to_dict()}")
    return train, test_clean, zero_info


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default=RAW_DIR)
    p.add_argument("--out-dir", default=OUT_DIR)
    p.add_argument("--val-days", type=int, default=16)
    p.add_argument("--expected-series", type=int, default=1782)
    p.add_argument("--zero-tail-days", type=int, default=28)
    a = p.parse_args()
    run(a.raw_dir, a.out_dir, a.val_days, a.expected_series, a.zero_tail_days)
