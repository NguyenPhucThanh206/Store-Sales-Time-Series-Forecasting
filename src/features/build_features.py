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
                       để ánh xạ về thứ tự test.csv khi xuất submission, và cờ `is_zero_series`.
    Mục tiêu: TARGET = "sales_log1p". Danh sách đặc trưng: FEATURES, cột phân loại: CAT_FEATURES.

ĐỌC BẰNG `load_fe()`, không dùng pd.read_parquet trực tiếp:
    parquet (pyarrow) không lưu dtype `category` cho cột số nguyên -> store_nbr / cluster đọc lại
    thành int (XGBoost enable_categorical sẽ coi là biến số). `load_fe()` ép mọi cột CAT_FEATURES
    về `category` với cùng danh mục ở train và test.

        from src.features.build_features import load_fe, FEATURES, CAT_FEATURES, TARGET
        train_fe, test_fe = load_fe()

BASE_FEATURES giữ 38 đặc trưng T05 gốc. FEATURES bổ sung các nhóm trong
guided_features.FEATURE_GROUPS; check_no_leakage kiểm tra cả hai bộ. Các
đặc trưng giao dịch dùng transactions.csv tại thời điểm t-16 hoặc sớm hơn.

Quy ước thứ tự dòng (bàn giao cho T06-T08, T13):
    * Dự báo test (.npy, thang log1p) lưu THEO THỨ TỰ DÒNG CỦA test_fe = (series_id, date), tức
      1.782 chuỗi x 16 ngày (reshape được thành (1782, 16), cùng thứ tự với GRU).
    * `id` của test_fe KHÔNG tăng dần -> chỉ ánh xạ về thứ tự test.csv khi ghi submission
      (`sub = pd.DataFrame({"id": test_fe["id"], "sales": ...}).sort_values("id")`).
    * Dự báo Validation (.npy) theo thứ tự dòng của train_fe[train_fe.is_val == 1].

Quy tắc chống rò rỉ (horizon 16 ngày, một mô hình cho cả 16 ngày):
    * Mọi đặc trưng từ doanh số chỉ dùng giá trị tại t-16 trở về trước (lag >= 16, rolling trên
      chuỗi đã dịch 16 ngày) -> với ngày test xa nhất (2017-08-31) vẫn chỉ cần dữ liệu <= 2017-08-15.
    * onpromotion, lịch, ngày lễ, giá dầu là thông tin biết trước cho cả giai đoạn test.
      Riêng giá dầu giai đoạn test chỉ "biết trước" vì oil.csv của Kaggle phủ tới 2017-08-31;
      trong thực tế phải dùng giá dầu trễ >= 16 ngày (ghi chú này cần có trong báo cáo).
    * Kiểm tra tự động: dựng lại đặc trưng với doanh số (`sales` và `sales_log1p`) 16 ngày
      Validation bị xoá (NaN) -> đặc trưng của các dòng Validation phải không đổi
      (`check_no_leakage`); lag / rolling được đối chiếu theo ngày trên cả train lẫn test
      (`check_lag_alignment`).

Chuỗi toàn 0 (53 chuỗi, bị T02 loại khỏi train) chỉ có trong test_fe (`is_zero_series == 1`), mọi
đặc trưng doanh số = NaN -> mô hình sẽ dự báo khác 0. Hậu xử lý BẮT BUỘC ép dự báo của chúng về 0
(`apply_zero_series`), nên assert lại ở bước blend / submission.
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from .guided_features import GUIDED_FEATURES, add_guided_features

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"

HORIZON = 16
SALES_LAGS = (16, 21, 28, 364)
ROLL_WINDOWS = (7, 28)
ROLL_MIN_PERIODS = {7: 4, 28: 14}  # cần >= nửa cửa sổ, tránh trung bình chỉ từ 1-2 ngày ở chuỗi mới
MAX_HOLIDAY_DIST = 60

TARGET = "sales_log1p"
KEYS = ["series_id", "date", "store_nbr", "family"]
CAT_FEATURES = ["store_nbr", "family", "city", "state", "store_type", "cluster"]
HOLIDAY_FEATURES = [
    "hol_national", "hol_regional", "hol_local", "is_holiday", "is_bridge", "is_additional",
    "is_workday", "is_event", "is_earthquake", "is_shopping_event", "is_sport_event",
]
BASE_FEATURES = [
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
FEATURES = [*BASE_FEATURES, *GUIDED_FEATURES]


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
        "transactions": pd.read_csv(raw_dir / "transactions.csv"),
        "zero": (pd.read_csv(proc_dir / "zero_series.csv") if (proc_dir / "zero_series.csv").exists()
                 else None),
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
    # Giữ day_of_year gốc (sau 29/2/2016 lệch +1 so với năm thường). Đã thử bỏ 29/2 khỏi thang đếm:
    # val RMSLE xấu đi ~0,004 (LightGBM smoke, 2 seed); bỏ hẳn dayofyear còn xấu hơn (~0,008).
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
    # Ngày test cuối (2017-08-31) không có ngày kế tiếp -> lấp bằng khuyến mãi cùng ngày, để không
    # sinh ra NaN mà mô hình chưa từng thấy lúc train
    df["onpromotion_lead1"] = g.shift(-1).fillna(df["onpromotion"]).astype("float32")
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
            lambda s, w=w: s.rolling(w, min_periods=ROLL_MIN_PERIODS[w]).mean()).astype("float32")
    df["rolling_std_28"] = gb.transform(
        lambda s: s.rolling(28, min_periods=ROLL_MIN_PERIODS[28]).std()).astype("float32")
    return df


def to_categorical(df: pd.DataFrame) -> pd.DataFrame:
    """Ép mọi cột CAT_FEATURES sang `category` (danh mục sắp xếp) sau khi đã merge xong.

    store_nbr / cluster là số nguyên nhưng không có thứ tự ý nghĩa; XGBoost enable_categorical
    chỉ coi dtype `category` là biến phân loại.
    """
    for c in CAT_FEATURES:
        if not isinstance(df[c].dtype, pd.CategoricalDtype):
            df[c] = pd.Categorical(df[c], categories=np.sort(df[c].unique()))
    return df


def build(full: pd.DataFrame, stores, oil, cal, transactions) -> pd.DataFrame:
    df = add_calendar(full.copy())
    df = add_external(df, stores, oil, cal)
    df = add_promo(df)
    df = add_sales_lags(df)
    df = to_categorical(df)
    df = add_guided_features(df, transactions)
    return df.sort_values(["series_id", "date"]).reset_index(drop=True)


def align_categories(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Ép CAT_FEATURES của train/test về `category` với CÙNG danh mục (sắp xếp), sửa tại chỗ."""
    for c in CAT_FEATURES:
        cats = pd.Index(np.union1d(train[c].astype(object).dropna().unique(),
                                   test[c].astype(object).dropna().unique()))
        for d in (train, test):
            d[c] = pd.Categorical(d[c].astype(object), categories=cats)


def load_fe(proc_dir=PROC_DIR, columns=None):
    """Đọc (train_fe, test_fe) và khôi phục dtype `category` cho CAT_FEATURES."""
    proc_dir = Path(proc_dir)
    cols_tr = cols_te = None
    if columns is not None:
        keep = list(dict.fromkeys([*columns, *CAT_FEATURES]))
        cols_tr = [c for c in keep if c != "id"]
        cols_te = [c for c in keep if c not in ("sales", TARGET, "is_val", "is_closed")]
    train = pd.read_parquet(proc_dir / "train_fe.parquet", columns=cols_tr)
    test = pd.read_parquet(proc_dir / "test_fe.parquet", columns=cols_te)
    align_categories(train, test)
    return train, test


def apply_zero_series(pred_log1p: np.ndarray, test_fe: pd.DataFrame) -> np.ndarray:
    """Ép dự báo (thang log1p, theo thứ tự dòng test_fe) của 53 chuỗi toàn 0 về 0."""
    pred = np.asarray(pred_log1p, dtype="float64").copy()
    assert len(pred) == len(test_fe), "Dự báo không cùng số dòng với test_fe"
    pred[test_fe["is_zero_series"].to_numpy() == 1] = 0.0
    return pred


# ------------------------------------------------------------------- checks
def check_no_leakage(full, stores, oil, cal, transactions, built) -> None:
    """Xoá doanh số 16 ngày Validation rồi dựng lại: đặc trưng các dòng Validation phải y hệt.

    Xoá cả `sales` gốc lẫn `sales_log1p` để bắt được cả đặc trưng sau này tính từ `sales`.
    """
    masked = full.copy()
    masked.loc[masked["is_val"] == 1, ["sales", TARGET]] = np.nan
    rebuilt = build(masked, stores, oil, cal, transactions)
    val = built["is_val"] == 1
    a = built.loc[val, FEATURES].reset_index(drop=True)
    b = rebuilt.loc[val, FEATURES].reset_index(drop=True)
    pd.testing.assert_frame_equal(a, b)


def check_lag_alignment(built: pd.DataFrame, n_samples: int = 2000, seed: int = 0) -> None:
    """Đối chiếu lag / rolling với giá trị tra theo NGÀY (không theo vị trí dòng).

    sales_lag_k(d) = sales_log1p(d-k); rolling_mean_w(d) = trung bình sales_log1p trên
    [d-16-w+1, d-16] (bỏ NaN, cần >= ROLL_MIN_PERIODS[w] giá trị). Kiểm trên một mẫu train và
    TOÀN BỘ dòng test.
    """
    look = built.set_index(["series_id", "date"])[TARGET]
    rows = pd.concat([built[built["is_test"] == 0].sample(n_samples, random_state=seed),
                      built[built["is_test"] == 1]])
    sid, date = rows["series_id"].to_numpy(), rows["date"]

    def lookup(offset_days: int) -> np.ndarray:
        key = pd.MultiIndex.from_arrays([sid, date - pd.Timedelta(days=offset_days)])
        return look.reindex(key).to_numpy(dtype="float64")

    for k in SALES_LAGS:
        np.testing.assert_allclose(rows[f"sales_lag_{k}"].to_numpy(), lookup(k).astype("float32"),
                                   equal_nan=True, err_msg=f"sales_lag_{k}")
    for w in ROLL_WINDOWS:
        win = np.column_stack([lookup(HORIZON + j) for j in range(w)])
        cnt = np.isfinite(win).sum(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # nanmean của hàng toàn NaN
            expect = np.where(cnt >= ROLL_MIN_PERIODS[w], np.nanmean(win, axis=1), np.nan)
        np.testing.assert_allclose(rows[f"rolling_mean_{w}"].to_numpy(), expect.astype("float32"),
                                   rtol=1e-5, atol=1e-5, equal_nan=True,
                                   err_msg=f"rolling_mean_{w}")


# --------------------------------------------------------------------------- run
def run(raw_dir=RAW_DIR, proc_dir=PROC_DIR, check=True):
    raw_dir, proc_dir = Path(raw_dir), Path(proc_dir)
    inp = load_inputs(raw_dir, proc_dir)
    full = stack_train_test(inp["train"], inp["test"])
    df = build(full, inp["stores"], inp["oil"], inp["cal"], inp["transactions"])

    if check:
        check_lag_alignment(df)
        check_no_leakage(full, inp["stores"], inp["oil"], inp["cal"], inp["transactions"], df)
        print("checks          : lag/rolling khớp theo ngày (train + test), "
              "không rò rỉ doanh số Validation -> OK")

    train_fe = df[df["is_test"] == 0].drop(columns=["id", "is_test"]).reset_index(drop=True)
    test_fe = df[df["is_test"] == 1].drop(
        columns=["sales", TARGET, "is_val", "is_closed", "is_test"]).reset_index(drop=True)
    test_fe["id"] = test_fe["id"].astype("int64")
    # Chỉ loại "all_zero" (bị T02 bỏ khỏi train); "zero_last_28d" vẫn train bình thường
    zero = inp["zero"]
    zero_ids = set() if zero is None else set(zero.loc[zero["kind"] == "all_zero", "series_id"])
    test_fe["is_zero_series"] = test_fe["series_id"].isin(zero_ids).astype("int8")

    assert len(train_fe) == len(inp["train"]) and len(test_fe) == len(inp["test"])
    assert set(test_fe["id"]) == set(inp["test"]["id"])
    assert not train_fe["series_id"].isin(zero_ids).any(), "Chuỗi toàn 0 lọt vào train"
    assert test_fe[[c for c in BASE_FEATURES if not c.startswith(("sales_", "rolling_"))]
                   ].notna().all().all(), "Đặc trưng không phải doanh số có NaN ở test"
    for c in CAT_FEATURES:
        assert isinstance(train_fe[c].dtype, pd.CategoricalDtype)
        assert train_fe[c].cat.categories.equals(test_fe[c].cat.categories), c

    train_fe.to_parquet(proc_dir / "train_fe.parquet", index=False)
    test_fe.to_parquet(proc_dir / "test_fe.parquet", index=False)

    # Đọc lại qua load_fe: dtype / danh mục phải y hệt bản trong bộ nhớ
    tr2, te2 = load_fe(proc_dir, columns=CAT_FEATURES)
    for c in CAT_FEATURES:
        assert tr2[c].cat.categories.equals(train_fe[c].cat.categories), c
        assert (tr2[c].cat.codes.to_numpy() == train_fe[c].cat.codes.to_numpy()).all(), c
        assert (te2[c].cat.codes.to_numpy() == test_fe[c].cat.codes.to_numpy()).all(), c

    val = train_fe[train_fe["is_val"] == 1]
    print(f"train_fe        : {len(train_fe):,} dòng x {train_fe.shape[1]} cột "
          f"({len(val):,} dòng Validation)")
    print(f"test_fe         : {len(test_fe):,} dòng x {test_fe.shape[1]} cột")
    print(f"features        : {len(FEATURES)} ({len(CAT_FEATURES)} phân loại)")
    print(f"chuỗi toàn 0    : {test_fe['is_zero_series'].sum() // HORIZON} chuỗi trong test")
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
