"""
Owner: An Đức Minh Trí (24021645) - Task T05
Bộ Feature Engineering dùng chung cho nhánh Non-DL (LightGBM / CatBoost / XGBoost).

    python -m src.preprocess.run_preprocess   # T02 -> train_clean / test_clean
    python -m src.features.external           # T03 -> oil_clean / holidays_store
    python -m src.features.build_features     # T05 -> train_fe / test_fe

Đầu ra (data/processed/):
    train_fe.parquet - mọi dòng của train_clean (gồm cả 16 ngày Validation, cột is_val),
                       sắp theo (series_id, date) = (store_nbr, family, date), giống T02 và GRU.
    test_fe.parquet  - mọi dòng của test.csv (28.512), sắp theo (series_id, date), giữ cột `id`
                       để ánh xạ về thứ tự test.csv khi xuất submission.
    Mục tiêu: TARGET = "sales_log1p". Danh sách đặc trưng: FEATURES, cột phân loại: CAT_FEATURES.

Quy tắc chống rò rỉ (horizon 16 ngày, một mô hình cho cả 16 ngày):
    * Mọi đặc trưng từ doanh số chỉ dùng giá trị tại t-16 trở về trước (lag >= 16, rolling trên
      chuỗi đã dịch 16 ngày) -> với ngày test xa nhất (2017-08-31) vẫn chỉ cần dữ liệu <= 2017-08-15.
    * onpromotion, lịch, ngày lễ, giá dầu là thông tin biết trước cho cả giai đoạn test.
    * Kiểm tra tự động: dựng lại đặc trưng với doanh số 16 ngày Validation bị xoá (NaN) -> đặc trưng
      của các dòng Validation phải không đổi (xem `check_no_leakage`).

Chuỗi toàn 0 (53 chuỗi, bị T02 loại khỏi train) chỉ có trong test_fe, mọi đặc trưng doanh số = NaN;
hậu xử lý ép dự báo của chúng về 0 (zero_series.csv).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"

HORIZON = 16
SALES_LAGS = (16, 21, 28, 364)
ROLL_WINDOWS = (7, 28)
MAX_HOLIDAY_DIST = 60

TARGET = "sales_log1p"
KEYS = ["series_id", "date", "store_nbr", "family"]
CAT_FEATURES = ["store_nbr", "family", "city", "state", "store_type", "cluster"]
HOLIDAY_FEATURES = [
    "hol_national", "hol_regional", "hol_local", "is_holiday", "is_bridge", "is_additional",
    "is_workday", "is_event", "is_earthquake", "is_shopping_event", "is_sport_event",
]
FEATURES = [
    *CAT_FEATURES,
    # thời gian & ngày lương
    "dayofweek", "day_of_month", "month", "dayofyear", "is_payday", "days_since_payday",
    # ngày lễ (đã khớp địa lý theo cửa hàng)
    *HOLIDAY_FEATURES, "days_to_holiday", "days_since_holiday",
    # giá dầu
    "oil", "oil_ma7", "oil_ma14",
    # khuyến mãi
    "onpromotion", "onpromotion_ma7", "onpromotion_lead1",
    # doanh số quá khứ (thang log1p, an toàn horizon 16)
    *(f"sales_lag_{k}" for k in SALES_LAGS),
    *(f"rolling_mean_{w}" for w in ROLL_WINDOWS),
    "rolling_std_28",
]


# ------------------------------------------------------------------ load inputs
def load_inputs(raw_dir: Path, proc_dir: Path) -> dict:
    need = ["train_clean.parquet", "test_clean.parquet", "oil_clean.parquet", "holidays_store.parquet"]
    missing = [f for f in need if not (proc_dir / f).exists()]
    if missing:
        raise FileNotFoundError(f"Thiếu {missing} trong {proc_dir}: chạy T02 (src.preprocess."
                                "run_preprocess) và T03 (src.features.external) trước")
    return {
        "train": pd.read_parquet(proc_dir / "train_clean.parquet"),
        "test": pd.read_parquet(proc_dir / "test_clean.parquet"),
        "oil": pd.read_parquet(proc_dir / "oil_clean.parquet"),
        "cal": pd.read_parquet(proc_dir / "holidays_store.parquet"),
        "stores": pd.read_csv(raw_dir / "stores.csv"),
    }


def stack_train_test(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Nối train + test thành một bảng dài liên tục theo ngày cho từng chuỗi."""
    tr = train[KEYS + ["onpromotion", "sales", TARGET, "is_closed", "is_val"]].copy()
    tr["is_test"] = np.int8(0)
    te = test[KEYS + ["id", "onpromotion"]].copy()
    te["is_test"] = np.int8(1)
    te["is_val"] = np.int8(0)
    te["is_closed"] = np.int8(0)
    full = pd.concat([tr, te], ignore_index=True)
    full["family"] = full["family"].astype(str).astype("category")
    full["date"] = pd.to_datetime(full["date"])
    full = full.sort_values(["series_id", "date"]).reset_index(drop=True)

    gap = full.groupby("series_id")["date"].diff().dropna()
    if not (gap == pd.Timedelta(days=1)).all():
        raise AssertionError("Chuỗi không liên tục theo ngày -> shift theo dòng sẽ lệch lịch")
    return full


# ------------------------------------------------------------------- features
def add_calendar(df: pd.DataFrame) -> pd.DataFrame:
    d = df["date"].dt
    df["dayofweek"] = d.day_of_week.astype("int8")
    df["day_of_month"] = d.day.astype("int8")
    df["month"] = d.month.astype("int8")
    df["dayofyear"] = d.day_of_year.astype("int16")
    df["is_payday"] = ((d.day == 15) | d.is_month_end).astype("int8")
    # Số ngày kể từ lần nhận lương gần nhất (ngày 15 / ngày cuối tháng trước); EDA T03: đỉnh ở ngày 1-3
    df["days_since_payday"] = np.where(d.is_month_end, 0,
                                       np.where(d.day >= 15, d.day - 15, d.day)).astype("int8")
    return df


def holiday_distances(cal: pd.DataFrame, max_dist: int = MAX_HOLIDAY_DIST) -> pd.DataFrame:
    """days_to_holiday / days_since_holiday theo từng cửa hàng (is_holiday đã khớp địa lý).

    Tính trên toàn bộ lịch holidays_store (rộng hơn train/test) rồi cắt ở max_dist.
    """
    cal = cal.sort_values(["store_nbr", "date"])
    hday = cal["date"].where(cal["is_holiday"] == 1)
    nxt = hday.groupby(cal["store_nbr"]).bfill()
    prv = hday.groupby(cal["store_nbr"]).ffill()
    out = cal[["date", "store_nbr"]].copy()
    out["days_to_holiday"] = (nxt - cal["date"]).dt.days.clip(upper=max_dist).fillna(max_dist)
    out["days_since_holiday"] = (cal["date"] - prv).dt.days.clip(upper=max_dist).fillna(max_dist)
    return out.astype({"days_to_holiday": "int8", "days_since_holiday": "int8"})


def add_external(df, stores, oil, cal) -> pd.DataFrame:
    st = stores.rename(columns={"type": "store_type"})[
        ["store_nbr", "city", "state", "store_type", "cluster"]].copy()
    st["store_nbr"] = st["store_nbr"].astype(df["store_nbr"].dtype)
    for c in ("city", "state", "store_type"):
        st[c] = st[c].astype("category")
    st["cluster"] = st["cluster"].astype("int8")

    cal = cal.copy()
    cal["store_nbr"] = cal["store_nbr"].astype(df["store_nbr"].dtype)
    cal = cal.merge(holiday_distances(cal), on=["date", "store_nbr"], how="left")
    cal = cal[["date", "store_nbr", *HOLIDAY_FEATURES, "days_to_holiday", "days_since_holiday"]]

    n = len(df)
    df = (df.merge(st, on="store_nbr", how="left")
            .merge(oil[["date", "oil", "oil_ma7", "oil_ma14"]], on="date", how="left")
            .merge(cal, on=["date", "store_nbr"], how="left"))
    assert len(df) == n, "Merge làm thay đổi số dòng"
    for c in ["city", "oil", "is_holiday", "days_to_holiday"]:
        assert df[c].notna().all(), f"Cột {c} có NaN sau merge"
    return df


def add_promo(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("series_id", sort=False)["onpromotion"]
    df["onpromotion_ma7"] = g.transform(lambda s: s.rolling(7, min_periods=1).mean()).astype("float32")
    df["onpromotion_lead1"] = g.shift(-1).astype("float32")
    return df


def add_sales_lags(df: pd.DataFrame, horizon: int = HORIZON) -> pd.DataFrame:
    g = df.groupby("series_id", sort=False)[TARGET]
    for k in SALES_LAGS:
        assert k >= horizon, f"lag {k} < horizon {horizon} -> rò rỉ"
        df[f"sales_lag_{k}"] = g.shift(k).astype("float32")
    base = g.shift(horizon)  # chỉ dùng dữ liệu <= t-16
    gb = base.groupby(df["series_id"], sort=False)
    for w in ROLL_WINDOWS:
        df[f"rolling_mean_{w}"] = gb.transform(
            lambda s, w=w: s.rolling(w, min_periods=1).mean()).astype("float32")
    df["rolling_std_28"] = gb.transform(
        lambda s: s.rolling(28, min_periods=2).std()).astype("float32")
    return df


def build(full: pd.DataFrame, stores, oil, cal) -> pd.DataFrame:
    df = add_calendar(full.copy())
    df = add_external(df, stores, oil, cal)
    df = add_promo(df)
    df = add_sales_lags(df)
    return df.sort_values(["series_id", "date"]).reset_index(drop=True)


# ------------------------------------------------------------------- checks
def check_no_leakage(full, stores, oil, cal, built) -> None:
    """Xoá doanh số 16 ngày Validation rồi dựng lại: đặc trưng các dòng Validation phải y hệt."""
    masked = full.copy()
    masked.loc[masked["is_val"] == 1, TARGET] = np.nan
    rebuilt = build(masked, stores, oil, cal)
    val = built["is_val"] == 1
    a = built.loc[val, FEATURES].reset_index(drop=True)
    b = rebuilt.loc[val, FEATURES].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def check_lag_alignment(built: pd.DataFrame, n_samples: int = 2000, seed: int = 0) -> None:
    """sales_lag_k tại ngày d phải bằng sales_log1p tra theo ngày d-k (không theo vị trí dòng)."""
    look = built.set_index(["series_id", "date"])[TARGET]
    rows = built[built["is_test"] == 0].sample(n_samples, random_state=seed)
    for k in SALES_LAGS:
        key = pd.MultiIndex.from_arrays([rows["series_id"], rows["date"] - pd.Timedelta(days=k)])
        expect = look.reindex(key).to_numpy(dtype="float32")
        np.testing.assert_allclose(rows[f"sales_lag_{k}"].to_numpy(), expect, equal_nan=True)


# --------------------------------------------------------------------------- run
def run(raw_dir=RAW_DIR, proc_dir=PROC_DIR, check=True):
    raw_dir, proc_dir = Path(raw_dir), Path(proc_dir)
    inp = load_inputs(raw_dir, proc_dir)
    full = stack_train_test(inp["train"], inp["test"])
    df = build(full, inp["stores"], inp["oil"], inp["cal"])

    if check:
        check_lag_alignment(df)
        check_no_leakage(full, inp["stores"], inp["oil"], inp["cal"], df)
        print("checks          : lag khớp theo ngày, không rò rỉ doanh số Validation -> OK")

    train_fe = df[df["is_test"] == 0].drop(columns=["id", "is_test"]).reset_index(drop=True)
    test_fe = df[df["is_test"] == 1].drop(
        columns=["sales", TARGET, "is_val", "is_closed", "is_test"]).reset_index(drop=True)
    test_fe["id"] = test_fe["id"].astype("int64")

    assert len(train_fe) == len(inp["train"]) and len(test_fe) == len(inp["test"])
    assert set(test_fe["id"]) == set(inp["test"]["id"])
    assert test_fe[[c for c in FEATURES if not c.startswith(("sales_", "rolling_", "onpromotion_lead"))]
                   ].notna().all().all()

    train_fe.to_parquet(proc_dir / "train_fe.parquet", index=False)
    test_fe.to_parquet(proc_dir / "test_fe.parquet", index=False)

    val = train_fe[train_fe["is_val"] == 1]
    print(f"train_fe        : {len(train_fe):,} dòng x {train_fe.shape[1]} cột "
          f"({len(val):,} dòng Validation)")
    print(f"test_fe         : {len(test_fe):,} dòng x {test_fe.shape[1]} cột")
    print(f"features        : {len(FEATURES)} ({len(CAT_FEATURES)} phân loại)")
    na = test_fe[[c for c in FEATURES if c.startswith(("sales_", "rolling_"))]].isna().mean()
    print("tỉ lệ NaN (test):", na.round(4).to_dict())
    return train_fe, test_fe


if __name__ == "__main__":
    # Console Windows (cp1252) không in được tiếng Việt
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default=RAW_DIR)
    p.add_argument("--proc-dir", default=PROC_DIR)
    p.add_argument("--no-check", action="store_true", help="bỏ qua kiểm tra rò rỉ (nhanh hơn)")
    a = p.parse_args()
    run(a.raw_dir, a.proc_dir, check=not a.no_check)
