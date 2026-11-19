"""
Owner: An Đức Minh Trí (24021645) - Task T03
Xử lý dữ liệu ngoại sinh: giá dầu (oil.csv) và ngày lễ (holidays_events.csv + stores.csv).

    python -m src.features.external
    python -m src.features.external --raw-dir data/raw --out-dir data/processed

Đầu ra (data/processed/), phủ toàn bộ lịch train + test (2013-01-01 -> 2017-08-31):
    oil_clean.parquet         - date, dcoilwtico (gốc, có NaN), oil, oil_ma7, oil_ma14
    holidays_store.parquet    - lưới đầy đủ (date x store_nbr), các cờ ngày lễ đã khớp địa lý

Quy tắc ngày lễ:
    * transferred == True    -> loại (ngày lễ đã dời đi, hôm đó là ngày làm việc bình thường)
    * type == "Transfer"     -> giữ, là ngày nghỉ thực tế sau khi dời
    * Holiday / Transfer / Additional / Bridge -> ngày nghỉ (is_holiday)
    * "Work Day"             -> ngày bù làm việc (thường là thứ 7), cờ riêng is_workday
    * "Event"                -> sự kiện toàn quốc (động đất, World Cup, Black Friday...), cờ riêng is_event
    * National -> mọi cửa hàng; Regional -> khớp stores.state; Local -> khớp stores.city
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Console Windows (cp1252) không in được tiếng Việt
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"

HOLIDAY_TYPES = ("Holiday", "Transfer", "Additional", "Bridge")
LOCALE_KEY = {"National": None, "Regional": "state", "Local": "city"}

HOLIDAY_FLAGS = [
    "hol_national", "hol_regional", "hol_local", "is_holiday",
    "is_bridge", "is_additional", "is_workday", "is_event",
]


# --------------------------------------------------------------------------- oil
def prepare_oil(oil: pd.DataFrame, start, end) -> pd.DataFrame:
    """Lưới ngày liên tục [start, end]; nội suy tuyến tính giá dầu cho cuối tuần / ngày thiếu.

    Đầu/cuối chuỗi không nội suy được thì lấp bằng giá trị gần nhất.
    oil_ma7 / oil_ma14 là trung bình trượt lùi (chỉ dùng quá khứ + hôm nay).
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

    Trả về bảng dài (date, store_nbr, type, locale) - có thể trùng nhiều dòng / ngày.
    """
    st = stores[["store_nbr", "city", "state"]]
    parts = []
    for locale, key in LOCALE_KEY.items():
        h = hol.loc[hol["locale"] == locale, ["date", "type", "locale", "locale_name"]]
        if h.empty:
            continue
        if key is None:
            m = h.merge(st[["store_nbr"]], how="cross")
        else:
            m = h.merge(st[["store_nbr", key]], left_on="locale_name", right_on=key, how="inner")
        parts.append(m[["date", "store_nbr", "type", "locale"]])

    unknown = set(hol.loc[hol["locale"] != "National", "locale_name"]) - \
        set(st["city"]) - set(st["state"])
    if unknown:
        print(f"[holidays] locale_name không khớp cửa hàng nào (bỏ qua): {sorted(unknown)}")
    return pd.concat(parts, ignore_index=True)


def build_holiday_calendar(hol: pd.DataFrame, stores: pd.DataFrame, start, end) -> pd.DataFrame:
    """Lưới đầy đủ (date x store_nbr) với các cờ int8 trong HOLIDAY_FLAGS."""
    hol = clean_holidays(hol)
    m = match_holidays_to_stores(hol, stores)

    is_off = m["type"].isin(HOLIDAY_TYPES)
    m["hol_national"] = is_off & (m["locale"] == "National")
    m["hol_regional"] = is_off & (m["locale"] == "Regional")
    m["hol_local"] = is_off & (m["locale"] == "Local")
    m["is_holiday"] = is_off
    m["is_bridge"] = m["type"].eq("Bridge")
    m["is_additional"] = m["type"].eq("Additional")
    m["is_workday"] = m["type"].eq("Work Day")
    m["is_event"] = m["type"].eq("Event")
    flags = m.groupby(["date", "store_nbr"])[HOLIDAY_FLAGS].max()

    dates = pd.date_range(start, end, freq="D")
    idx = pd.MultiIndex.from_product(
        [dates, np.sort(stores["store_nbr"].unique())], names=["date", "store_nbr"])
    cal = flags.reindex(idx, fill_value=False).astype("int8").reset_index()
    cal["store_nbr"] = cal["store_nbr"].astype("int16")
    return cal


# --------------------------------------------------------------------------- run
def infer_date_range(raw_dir: Path):
    """Ngày đầu train -> ngày cuối test."""
    start = pd.read_csv(raw_dir / "train.csv", usecols=["date"], nrows=1)["date"].iloc[0]
    end = pd.read_csv(raw_dir / "test.csv", usecols=["date"])["date"].max()
    return pd.Timestamp(start), pd.Timestamp(end)


def run(raw_dir=RAW_DIR, out_dir=OUT_DIR):
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start, end = infer_date_range(raw_dir)
    oil = pd.read_csv(raw_dir / "oil.csv")
    hol = pd.read_csv(raw_dir / "holidays_events.csv")
    stores = pd.read_csv(raw_dir / "stores.csv")

    oil_clean = prepare_oil(oil, start, end)
    cal = build_holiday_calendar(hol, stores, start, end)

    # ---- kiểm tra trước khi bàn giao ----
    n_days = (end - start).days + 1
    assert len(oil_clean) == n_days and oil_clean["date"].is_unique
    assert len(cal) == n_days * stores["store_nbr"].nunique()
    assert not cal.duplicated(["date", "store_nbr"]).any()

    oil_clean.to_parquet(out_dir / "oil_clean.parquet", index=False)
    cal.to_parquet(out_dir / "holidays_store.parquet", index=False)

    n_missing = int(oil_clean["dcoilwtico"].isna().sum())
    print(f"range           : {start.date()} -> {end.date()} ({n_days} ngày)")
    print(f"oil_clean       : {len(oil_clean):,} dòng, nội suy {n_missing} ngày thiếu giá")
    print(f"holidays_store  : {len(cal):,} dòng ({stores['store_nbr'].nunique()} cửa hàng)")
    print("tỉ lệ (store, ngày) có cờ:", cal[HOLIDAY_FLAGS].mean().round(4).to_dict())
    return oil_clean, cal


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default=RAW_DIR)
    p.add_argument("--out-dir", default=OUT_DIR)
    a = p.parse_args()
    run(a.raw_dir, a.out_dir)
