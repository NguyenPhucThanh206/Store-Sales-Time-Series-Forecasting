"""
Owner: An Đức Minh Trí (24021645) - Task T03
Xử lý dữ liệu ngoại sinh: giá dầu (oil.csv) và ngày lễ (holidays_events.csv + stores.csv).

    python -m src.features.external
    python -m src.features.external --raw-dir data/raw --out-dir data/processed

Đầu ra (data/processed/):
    oil_clean.parquet       - lịch train + test (2013-01-01 -> 2017-08-31):
                              date, dcoilwtico (gốc, có NaN), oil, oil_ma7, oil_ma14
    holidays_store.parquet  - lưới đầy đủ (date x store_nbr) phủ HỢP của lịch train/test và
                              toàn bộ holidays_events.csv (2012-03-02 -> 2017-12-26), để T05
                              tính được days_to_holiday / days_since_holiday ở đầu train và
                              cuối test. Cắt về phạm vi train/test khi merge.

Quy tắc ngày lễ:
    * transferred == True        -> loại (ngày lễ đã dời đi, hôm đó là ngày làm việc bình thường)
    * Holiday / Transfer / Bridge -> ngày nghỉ thực tế (is_holiday; hol_* theo phạm vi địa lý)
    * Additional                 -> ngày mua sắm quanh lễ (chủ yếu quanh Giáng sinh), cờ riêng
                                    is_additional, KHÔNG tính vào is_holiday
    * Work Day                   -> ngày bù làm việc (thường là thứ 7), cờ riêng is_workday
    * Event                      -> is_event (bất kỳ), tách thêm is_earthquake (Terremoto Manabi),
                                    is_shopping_event (Black Friday, Cyber Monday, Dia de la Madre),
                                    is_sport_event (Mundial)
    * National -> mọi cửa hàng; Regional -> khớp stores.state; Local -> khớp stores.city
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"

DAYOFF_TYPES = ("Holiday", "Transfer", "Bridge")
LOCALE_KEY = {"National": None, "Regional": "state", "Local": "city"}
EVENT_PATTERNS = {
    "is_earthquake": r"Terremoto",
    "is_shopping_event": r"Black Friday|Cyber Monday|Dia de la Madre",
    "is_sport_event": r"Mundial",
}

HOLIDAY_FLAGS = [
    "hol_national", "hol_regional", "hol_local", "is_holiday",
    "is_bridge", "is_additional", "is_workday",
    "is_event", *EVENT_PATTERNS,
]


# --------------------------------------------------------------------------- oil
def prepare_oil(oil: pd.DataFrame, start, end) -> pd.DataFrame:
    """Lưới ngày liên tục [start, end]; nội suy tuyến tính giá dầu cho cuối tuần / ngày thiếu.

    Đầu/cuối chuỗi không nội suy được thì lấp bằng giá trị gần nhất.
    oil_ma7 / oil_ma14 là trung bình trượt lùi (chỉ dùng quá khứ + hôm nay).
    Lưu ý: nội suy ngày cuối tuần dùng giá thứ 2 kế tiếp (rò rỉ <= 2 ngày). Với backtest
    nghiêm ngặt horizon 16 ngày, T05 nên dùng giá dầu trễ >= 16 ngày.
    """
    oil = oil[["date", "dcoilwtico"]].copy()
    oil["date"] = pd.to_datetime(oil["date"])
    oil = oil.drop_duplicates("date").set_index("date").sort_index()

    dates = pd.date_range(start, end, freq="D", name="date")
    out = oil.reindex(dates)
    out["oil"] = (out["dcoilwtico"].interpolate(method="time", limit_area="inside")
                  .ffill().bfill())
    out["oil_ma7"] = out["oil"].rolling(7, min_periods=1).mean()
    out["oil_ma14"] = out["oil"].rolling(14, min_periods=1).mean()
    out = out.astype("float32").reset_index()

    if out["oil"].isna().any():
        raise AssertionError("Giá dầu còn NaN sau nội suy")
    return out


# ---------------------------------------------------------------------- holidays
def clean_holidays(hol: pd.DataFrame) -> pd.DataFrame:
    """Loại các ngày lễ đã bị dời (transferred == True)."""
    hol = hol.copy()
    hol["date"] = pd.to_datetime(hol["date"])
    transferred = hol["transferred"].astype(str).str.lower().eq("true")
    return hol.loc[~transferred].reset_index(drop=True)


def match_holidays_to_stores(hol: pd.DataFrame, stores: pd.DataFrame) -> pd.DataFrame:
    """Nhân bản mỗi dòng ngày lễ sang các cửa hàng chịu ảnh hưởng theo phạm vi địa lý.

    Trả về bảng dài (date, store_nbr, type, locale, description) - có thể trùng nhiều dòng / ngày.
    """
    st = stores[["store_nbr", "city", "state"]]
    cols = ["date", "type", "locale", "locale_name", "description"]
    parts = []
    for locale, key in LOCALE_KEY.items():
        h = hol.loc[hol["locale"] == locale, cols]
        if h.empty:
            continue
        if key is None:
            m = h.merge(st[["store_nbr"]], how="cross")
        else:
            unknown = set(h["locale_name"]) - set(st[key])
            if unknown:
                print(f"[holidays] {locale}: locale_name không khớp stores.{key} "
                      f"(bỏ qua): {sorted(unknown)}")
            m = h.merge(st[["store_nbr", key]], left_on="locale_name", right_on=key, how="inner")
        parts.append(m[["date", "store_nbr", "type", "locale", "description"]])
    return pd.concat(parts, ignore_index=True)


def build_holiday_calendar(hol: pd.DataFrame, stores: pd.DataFrame, start, end) -> pd.DataFrame:
    """Lưới đầy đủ (date x store_nbr) trong [start, end] với các cờ int8 trong HOLIDAY_FLAGS."""
    hol = clean_holidays(hol)
    m = match_holidays_to_stores(hol, stores)

    dayoff = m["type"].isin(DAYOFF_TYPES)
    m["hol_national"] = dayoff & (m["locale"] == "National")
    m["hol_regional"] = dayoff & (m["locale"] == "Regional")
    m["hol_local"] = dayoff & (m["locale"] == "Local")
    m["is_holiday"] = dayoff
    m["is_bridge"] = m["type"].eq("Bridge")
    m["is_additional"] = m["type"].eq("Additional")
    m["is_workday"] = m["type"].eq("Work Day")
    event = m["type"].eq("Event")
    m["is_event"] = event
    for flag, pattern in EVENT_PATTERNS.items():
        m[flag] = event & m["description"].str.contains(pattern, regex=True)
    flags = m.groupby(["date", "store_nbr"])[HOLIDAY_FLAGS].max()

    dates = pd.date_range(start, end, freq="D")
    idx = pd.MultiIndex.from_product(
        [dates, np.sort(stores["store_nbr"].unique())], names=["date", "store_nbr"])
    cal = flags.reindex(idx, fill_value=False).astype("int8").reset_index()
    cal["store_nbr"] = cal["store_nbr"].astype("int16")
    return cal


# --------------------------------------------------------------------------- run
def infer_date_range(raw_dir: Path):
    """Ngày đầu train -> ngày cuối test (không giả định file đã sắp xếp)."""
    start = pd.to_datetime(pd.read_csv(raw_dir / "train.csv", usecols=["date"])["date"]).min()
    end = pd.to_datetime(pd.read_csv(raw_dir / "test.csv", usecols=["date"])["date"]).max()
    return start, end


def run(raw_dir=RAW_DIR, out_dir=OUT_DIR):
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start, end = infer_date_range(raw_dir)
    oil = pd.read_csv(raw_dir / "oil.csv")
    hol = pd.read_csv(raw_dir / "holidays_events.csv")
    stores = pd.read_csv(raw_dir / "stores.csv")

    hol_dates = pd.to_datetime(hol["date"])
    cal_start, cal_end = min(start, hol_dates.min()), max(end, hol_dates.max())

    oil_clean = prepare_oil(oil, start, end)
    cal = build_holiday_calendar(hol, stores, cal_start, cal_end)

    # ---- kiểm tra trước khi bàn giao ----
    n_days = (end - start).days + 1
    n_cal_days = (cal_end - cal_start).days + 1
    assert len(oil_clean) == n_days and oil_clean["date"].is_unique
    assert len(cal) == n_cal_days * stores["store_nbr"].nunique()
    assert not cal.duplicated(["date", "store_nbr"]).any()
    assert cal_start <= start and cal_end >= end

    oil_clean.to_parquet(out_dir / "oil_clean.parquet", index=False)
    cal.to_parquet(out_dir / "holidays_store.parquet", index=False)

    n_missing = int(oil_clean["dcoilwtico"].isna().sum())
    in_range = cal[(cal["date"] >= start) & (cal["date"] <= end)]
    print(f"train/test range: {start.date()} -> {end.date()} ({n_days} ngày)")
    print(f"oil_clean       : {len(oil_clean):,} dòng, nội suy {n_missing} ngày thiếu giá")
    print(f"holidays_store  : {len(cal):,} dòng, {cal_start.date()} -> {cal_end.date()} "
          f"({n_cal_days} ngày x {stores['store_nbr'].nunique()} cửa hàng)")
    print("tỉ lệ (store, ngày) có cờ trong train/test:",
          in_range[HOLIDAY_FLAGS].mean().round(4).to_dict())
    return oil_clean, cal


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default=RAW_DIR)
    p.add_argument("--out-dir", default=OUT_DIR)
    a = p.parse_args()
    run(a.raw_dir, a.out_dir)
