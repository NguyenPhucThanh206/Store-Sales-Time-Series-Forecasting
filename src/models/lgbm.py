"""T06 - Huấn luyện LightGBM.

Chạy từ thư mục gốc repo:
    python -m src.models.lgbm            # chạy đầy đủ
    python -m src.models.lgbm --quick    # chạy nhanh để thử pipeline (vài phút)

Quy trình:
  1) Huấn luyện trên các dòng is_val == 0, early stopping + chấm RMSLE trên is_val == 1.
     Dự đoán Validation của mô hình này -> predictions/val/val_pred_lgbm.npy
  2) Huấn luyện lại trên TOÀN BỘ dữ liệu (gồm cả 16 ngày Validation) với số cây tốt nhất
     tìm được ở bước 1, rồi dự đoán Test -> predictions/test/test_pred_lgbm.npy

Cả hai file đều ở thang log1p. Thứ tự phần tử:
  - val : các dòng is_val == 1 của train_clean, sắp theo (series_id, date)
  - test: các dòng của test_clean, sắp theo (series_id, date)
Mục tiêu hồi quy MSE trên log1p chính là tối ưu RMSLE.
"""
import argparse
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC_DIR = ROOT / "data" / "processed"
PRED_DIR = ROOT / "predictions"
REPORT_DIR = ROOT / "reports"

META = {"id", "series_id", "date", "sales", "sales_log1p", "is_val", "is_closed"}
CAT_COLS = ["store_nbr", "family", "city", "state", "type"]


def rmsle(y_log, p_log) -> float:
    """Vì y, p đã ở thang log1p nên RMSLE = RMSE trên hai vector này."""
    return float(np.sqrt(np.mean((np.asarray(y_log) - np.asarray(p_log)) ** 2)))


def _check_order(df: pd.DataFrame, name: str):
    order = np.lexsort((df["date"].to_numpy(), df["series_id"].to_numpy()))
    assert (order == np.arange(len(df))).all(), f"{name} chưa sắp theo (series_id, date)"


def run(proc_dir=PROC_DIR, pred_dir=PRED_DIR, report_dir=REPORT_DIR,
        learning_rate=0.05, n_estimators=3000, num_leaves=127, seed=42, refit=True):
    proc_dir, pred_dir, report_dir = Path(proc_dir), Path(pred_dir), Path(report_dir)
    (pred_dir / "val").mkdir(parents=True, exist_ok=True)
    (pred_dir / "test").mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(proc_dir / "train_fe.parquet")
    test = pd.read_parquet(proc_dir / "test_fe.parquet")
    feats = [c for c in train.columns if c not in META]
    assert set(feats) == {c for c in test.columns if c not in META}, "train_fe và test_fe lệch cột"
    for c in CAT_COLS:  # đảm bảo cùng bảng mã danh mục
        test[c] = test[c].astype(train[c].dtype)
    test_x = test[feats]

    tr, va = train[train["is_val"] == 0], train[train["is_val"] == 1]
    _check_order(va, "validation")
    _check_order(test, "test_fe")
    print(f"train: {len(tr):,} dòng | validation: {len(va):,} dòng | test: {len(test):,} dòng | {len(feats)} đặc trưng")

    params = dict(
        objective="regression", metric="rmse", learning_rate=learning_rate,
        num_leaves=num_leaves, min_data_in_leaf=100, feature_fraction=0.7,
        bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, max_bin=255,
        seed=seed, n_jobs=-1, verbose=-1,
    )

    # ---- bước 1: huấn luyện, early stopping và chấm trên Validation
    d_tr = lgb.Dataset(tr[feats], tr["sales_log1p"], categorical_feature=CAT_COLS, free_raw_data=False)
    d_va = lgb.Dataset(va[feats], va["sales_log1p"], reference=d_tr)
    model = lgb.train(
        params, d_tr, num_boost_round=n_estimators, valid_sets=[d_va], valid_names=["val"],
        callbacks=[lgb.early_stopping(100), lgb.log_evaluation(100)],
    )
    best = int(model.best_iteration)
    val_pred = np.clip(model.predict(va[feats], num_iteration=best), 0, None).astype("float32")
    score = rmsle(va["sales_log1p"], val_pred)
    print(f"\n>>> Validation RMSLE = {score:.5f} (số cây tốt nhất: {best})")
    np.save(pred_dir / "val" / "val_pred_lgbm.npy", val_pred)

    imp = pd.DataFrame({"feature": feats, "gain": model.feature_importance("gain")})
    imp = imp.sort_values("gain", ascending=False)
    imp.to_csv(report_dir / "lgbm_feature_importance.csv", index=False)
    print("Top 10 đặc trưng:\n", imp.head(10).to_string(index=False))

    # ---- bước 2: huấn luyện lại trên toàn bộ dữ liệu rồi dự đoán Test
    if refit:
        d_full = lgb.Dataset(train[feats], train["sales_log1p"], categorical_feature=CAT_COLS)
        final = lgb.train(params, d_full, num_boost_round=max(50, int(best * 1.05)))
        test_pred = np.clip(final.predict(test_x), 0, None).astype("float32")
    else:
        test_pred = np.clip(model.predict(test_x, num_iteration=best), 0, None).astype("float32")
    np.save(pred_dir / "test" / "test_pred_lgbm.npy", test_pred)

    print(f"\nĐã lưu val_pred_lgbm.npy {val_pred.shape}, test_pred_lgbm.npy {test_pred.shape}")
    return score


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--learning-rate", type=float, default=0.05)
    p.add_argument("--n-estimators", type=int, default=3000)
    p.add_argument("--num-leaves", type=int, default=127)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-refit", action="store_true", help="Không huấn luyện lại trên toàn bộ dữ liệu")
    p.add_argument("--quick", action="store_true", help="Chạy nhanh (lr 0.1, 300 cây, 63 lá) để thử pipeline")
    a = p.parse_args()
    if a.quick:
        a.learning_rate, a.n_estimators, a.num_leaves = 0.1, 300, 63
    run(learning_rate=a.learning_rate, n_estimators=a.n_estimators, num_leaves=a.num_leaves,
        seed=a.seed, refit=not a.no_refit)
