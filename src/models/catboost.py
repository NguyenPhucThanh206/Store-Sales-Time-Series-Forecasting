"""
Owner: An Đức Minh Trí (24021645) - Task T08 (& T11)
Huấn luyện CatBoost trên bộ đặc trưng chung T05 (tuỳ chọn thêm nhóm guided của T07), thang log1p.

Chạy từ thư mục gốc repo, sau T02, T03, T05:
    python -m src.models.catboost              # mặc định: 38 đặc trưng T05, dữ liệu từ 2016, lr 0.1
    python -m src.models.catboost --quick      # thử pipeline nhanh (lr 0.2, tối đa 300 cây)
    python -m src.models.catboost --feature-set all --train-start all

Quy trình (giống T06):
  1) Huấn luyện trên các dòng is_val == 0, early stopping trên 16 ngày Validation (is_val == 1)
     -> predictions/val/val_pred_catboost.npy
  2) Huấn luyện lại trên TOÀN BỘ train_fe (gồm cả Validation) với số cây = best_iteration * 1.05
     -> predictions/test/test_pred_catboost.npy (đã ép 0 cho 53 chuỗi toàn 0)

Đầu ra:
  - val  : thứ tự dòng của train_fe[is_val == 1], sắp theo (series_id, date), 27.664 phần tử
  - test : thứ tự dòng của test_fe, sắp theo (series_id, date), 28.512 phần tử
  - cả hai ở thang log1p, cắt >= 0, float32
  - predictions/models/catboost.cbm (mô hình bước 2; cột phân loại được mã hoá bằng mã danh mục
    của load_fe), reports/catboost_feature_importance.csv, reports/catboost_results.md

Cột phân loại: CAT_FEATURES của T05 (store_nbr, family, city, state, store_type, cluster) khai
báo qua `cat_features` -> CatBoost tự học ordered target statistics + tổ hợp (vd store x family).
Lưu ý: early stopping dùng chính tập Validation nên val RMSLE hơi lạc quan (giống T06/T07).
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from src.features.build_features import (
    BASE_FEATURES, CAT_FEATURES, FEATURES, PROC_DIR, ROOT, TARGET, apply_zero_series, load_fe,
)

PRED_DIR = ROOT / "predictions"
REPORT_DIR = ROOT / "reports"
HORIZON = 16
N_ZERO_SERIES = 53

FEATURE_SETS = {"all": FEATURES, "base": BASE_FEATURES}
# Chọn theo thử nghiệm lr 0.1 (val RMSLE): base/2016 0.3977 | all/2016 0.4008 | all/2015 0.3996 |
# all/toàn bộ 2013+ kém rõ (lr 0.2: 0.4139 so với ~0.403). Xem reports/catboost_results.md.
DEFAULT_FEATURE_SET = "base"
DEFAULT_TRAIN_START = "2016-01-01"

PARAMS = dict(
    loss_function="RMSE",          # RMSE trên log1p(sales) chính là RMSLE
    eval_metric="RMSE",
    learning_rate=0.1,
    depth=8,
    l2_leaf_reg=3.0,
    border_count=254,
    one_hot_max_size=8,            # store_type (5 giá trị) -> one-hot; còn lại dùng CTR
    max_ctr_complexity=2,          # cho phép tổ hợp 2 cột phân loại (store_nbr x family ~ chuỗi)
    bootstrap_type="Bernoulli",
    subsample=0.8,
    random_seed=42,
    thread_count=-1,
    allow_writing_files=False,
)


def rmsle(y_log, p_log) -> float:
    """y, p đã ở thang log1p nên RMSLE = RMSE."""
    return float(np.sqrt(np.mean((np.asarray(y_log, "float64") - np.asarray(p_log, "float64")) ** 2)))


def kaggle_like_rmsle(y_log, p_log, n_zero_rows: int = N_ZERO_SERIES * HORIZON) -> float:
    """RMSLE khi tính thêm các chuỗi toàn 0 (bị T02 loại khỏi Validation) như dự báo 0 hoàn hảo."""
    se = (np.asarray(y_log, "float64") - np.asarray(p_log, "float64")) ** 2
    return float(np.sqrt(se.sum() / (len(se) + n_zero_rows)))


def check_inputs(train: pd.DataFrame, test: pd.DataFrame, feats: list) -> None:
    """Chặn dữ liệu sai thứ tự / thiếu cột / lệch danh mục trước khi huấn luyện."""
    for name, df, extra in (("train_fe", train, [TARGET, "is_val"]),
                            ("test_fe", test, ["id", "is_zero_series"])):
        missing = sorted(set(feats + extra + ["series_id", "date"]) - set(df.columns))
        if missing:
            raise ValueError(f"{name} thiếu cột {missing}: chạy lại src.features.build_features")
        order = np.lexsort((df["date"].to_numpy(), df["series_id"].to_numpy()))
        if not (order == np.arange(len(df))).all():
            raise ValueError(f"{name} chưa sắp theo (series_id, date)")
    val = train[train["is_val"] == 1]
    if val["date"].nunique() != HORIZON or val.groupby("series_id").size().ne(HORIZON).any():
        raise ValueError(f"Validation phải có đúng {HORIZON} ngày cho mỗi chuỗi")
    if test.groupby("series_id").size().ne(HORIZON).any():
        raise ValueError(f"Mỗi chuỗi test phải có đúng {HORIZON} ngày")
    for c in CAT_FEATURES:
        if not isinstance(train[c].dtype, pd.CategoricalDtype):
            raise TypeError(f"{c} phải có dtype category (đọc bằng load_fe)")
        if not train[c].cat.categories.equals(test[c].cat.categories):
            raise ValueError(f"Danh mục {c} khác nhau giữa train và test")
        if train[c].isna().any() or test[c].isna().any():
            raise ValueError(f"{c} có giá trị thiếu")


def to_pool(df: pd.DataFrame, feats: list, label=None) -> Pool:
    """CatBoost nhận cột phân loại dạng int/str: dùng mã danh mục (đã thống nhất train/test)."""
    x = df[feats].copy()
    cats = [c for c in feats if c in CAT_FEATURES]
    for c in cats:
        x[c] = x[c].cat.codes.astype("int32")
    return Pool(x, label=label, cat_features=cats)


def run(proc_dir=PROC_DIR, pred_dir=PRED_DIR, report_dir=REPORT_DIR, feature_set=DEFAULT_FEATURE_SET,
        train_start=DEFAULT_TRAIN_START, iterations=4000, learning_rate=None, depth=None,
        early_stopping=300, refit=True, seed=42, verbose=200) -> dict:
    proc_dir, pred_dir, report_dir = Path(proc_dir), Path(pred_dir), Path(report_dir)
    feats = list(FEATURE_SETS[feature_set])
    params = {**PARAMS, "random_seed": seed}
    if learning_rate is not None:
        params["learning_rate"] = learning_rate
    if depth is not None:
        params["depth"] = depth

    t0 = time.time()
    train, test = load_fe(proc_dir)
    check_inputs(train, test, feats)

    fit_mask = train[TARGET].notna() & (train["is_closed"] == 0)
    if train_start is not None:
        fit_mask &= train["date"] >= pd.Timestamp(train_start)
    tr = train[fit_mask & (train["is_val"] == 0)]
    full = train[fit_mask]
    va = train[train["is_val"] == 1]
    if va[TARGET].isna().any():
        raise ValueError("Validation có doanh số thiếu")
    print(f"train: {len(tr):,} dòng | validation: {len(va):,} | test: {len(test):,} | "
          f"{len(feats)} đặc trưng ({feature_set}), từ {tr['date'].min().date()}")

    # ---- bước 1: early stopping + chấm Validation
    model = CatBoostRegressor(iterations=iterations, od_type="Iter", od_wait=early_stopping,
                              use_best_model=True, **params)
    model.fit(to_pool(tr, feats, tr[TARGET]), eval_set=to_pool(va, feats, va[TARGET]),
              verbose=verbose)
    best = int(model.get_best_iteration()) + 1
    val_pred = np.clip(model.predict(to_pool(va, feats)), 0, None).astype("float32")
    score, score_kg = rmsle(va[TARGET], val_pred), kaggle_like_rmsle(va[TARGET], val_pred)
    t1 = time.time()
    print(f"\n>>> Validation RMSLE = {score:.5f} (tính cả 53 chuỗi toàn 0: {score_kg:.5f}), "
          f"best = {best} cây, {(t1 - t0) / 60:.1f} phút")

    imp = (pd.DataFrame({"feature": feats, "importance": model.get_feature_importance()})
           .sort_values("importance", ascending=False))

    # ---- bước 2: huấn luyện lại trên toàn bộ dữ liệu rồi dự báo Test
    if refit:
        n_refit = max(50, int(round(best * 1.05)))
        final = CatBoostRegressor(iterations=n_refit, **params)
        final.fit(to_pool(full, feats, full[TARGET]), verbose=verbose)
    else:
        n_refit, final = best, model
    test_pred = np.clip(final.predict(to_pool(test, feats)), 0, None)
    test_pred = apply_zero_series(test_pred, test).astype("float32")
    t2 = time.time()

    if len(val_pred) != len(va) or len(test_pred) != len(test):
        raise AssertionError("Số phần tử dự báo không khớp bảng đặc trưng")
    if not (np.isfinite(val_pred).all() and np.isfinite(test_pred).all()):
        raise ValueError("CatBoost sinh dự báo NaN/inf")

    paths = {"val": pred_dir / "val" / "val_pred_catboost.npy",
             "test": pred_dir / "test" / "test_pred_catboost.npy",
             "model": pred_dir / "models" / "catboost.cbm",
             "importance": report_dir / "catboost_feature_importance.csv",
             "report": report_dir / "catboost_results.md"}
    for p in paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
    np.save(paths["val"], val_pred)
    np.save(paths["test"], test_pred)
    final.save_model(str(paths["model"]))
    imp.to_csv(paths["importance"], index=False)

    result = {"val_rmsle": score, "val_rmsle_with_zero_series": score_kg, "best_iteration": best,
              "refit_iterations": n_refit, "feature_set": feature_set, "n_features": len(feats),
              "train_start": str(tr["date"].min().date()), "n_train_rows": len(tr),
              "minutes_step1": (t1 - t0) / 60, "minutes_total": (t2 - t0) / 60, "params": params}
    write_report(paths["report"], result, imp, val_pred, test_pred)
    print(f"Top 10 đặc trưng:\n{imp.head(10).to_string(index=False)}")
    print(f"\nĐã lưu {paths['val']} {val_pred.shape}, {paths['test']} {test_pred.shape}, "
          f"{paths['model'].name} ({(t2 - t0) / 60:.1f} phút)")
    return {**result, **{k: str(v) for k, v in paths.items()}}


def write_report(path: Path, r: dict, imp: pd.DataFrame, val_pred, test_pred) -> None:
    top = "\n".join(f"| {i} | `{f}` | {v:.2f} |" for i, (f, v) in
                    enumerate(zip(imp["feature"].head(15), imp["importance"].head(15)), 1))
    params = ", ".join(f"`{k}={v}`" for k, v in r["params"].items()
                       if k not in ("thread_count", "allow_writing_files"))
    path.write_text(f"""# Kết quả CatBoost (T08)

Sinh tự động bởi `python -m src.models.catboost`.

| Chỉ số | Giá trị |
| --- | --- |
| Validation RMSLE (27.664 dòng, 2017-07-31 → 08-15) | **{r['val_rmsle']:.5f}** |
| RMSLE tính cả 53 chuỗi toàn 0 (so với Kaggle) | {r['val_rmsle_with_zero_series']:.5f} |
| Số cây tốt nhất (early stopping) / huấn luyện lại | {r['best_iteration']} / {r['refit_iterations']} |
| Đặc trưng | {r['n_features']} (`{r['feature_set']}`) |
| Dữ liệu huấn luyện | {r['n_train_rows']:,} dòng, từ {r['train_start']} |
| Thời gian (bước 1 / tổng, CPU) | {r['minutes_step1']:.1f} / {r['minutes_total']:.1f} phút |
| Dự báo val / test (log1p, trung bình) | {val_pred.mean():.4f} / {test_pred.mean():.4f} |

Tham số: {params}.

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
{top}
""", encoding="utf-8")


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="T08 - CatBoost")
    p.add_argument("--feature-set", choices=sorted(FEATURE_SETS), default=DEFAULT_FEATURE_SET)
    p.add_argument("--train-start", default=DEFAULT_TRAIN_START,
                   help="Chỉ dùng dữ liệu từ ngày này; 'all' = toàn bộ từ 2013")
    p.add_argument("--iterations", type=int, default=4000)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--depth", type=int, default=None)
    p.add_argument("--early-stopping", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-refit", action="store_true", help="Không huấn luyện lại trên toàn bộ dữ liệu")
    p.add_argument("--quick", action="store_true", help="Thử pipeline: lr 0.2, tối đa 300 cây")
    a = p.parse_args()
    if a.quick:
        a.learning_rate, a.iterations = 0.2, 300
    run(feature_set=a.feature_set, train_start=None if a.train_start == "all" else a.train_start, iterations=a.iterations,
        learning_rate=a.learning_rate, depth=a.depth, early_stopping=a.early_stopping,
        refit=not a.no_refit, seed=a.seed)
