"""Đọc và kiểm tra schema 6 file CSV gốc trong data/raw/."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw"
OUT_DIR = ROOT / "data" / "processed"

REQUIRED_COLS = {
    "train": ["id", "date", "store_nbr", "family", "sales", "onpromotion"],
    "test": ["id", "date", "store_nbr", "family", "onpromotion"],
    "stores": ["store_nbr", "city", "state", "type", "cluster"],
    "oil": ["date", "dcoilwtico"],
    "holidays_events": ["date", "type", "locale", "locale_name", "description", "transferred"],
    "transactions": ["date", "store_nbr", "transactions"],
}


def load_raw(raw_dir: Path = RAW_DIR) -> dict:
    """Trả về dict {tên: DataFrame}. Báo lỗi rõ ràng nếu thiếu file / thiếu cột."""
    raw_dir = Path(raw_dir)
    tables = {}
    for name, cols in REQUIRED_COLS.items():
        path = raw_dir / f"{name}.csv"
        if not path.exists():
            raise FileNotFoundError(f"Thiếu file {path}")
        df = pd.read_csv(path)
        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f"{path.name} thiếu cột: {sorted(missing)}")
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        tables[name] = df

    # Ép kiểu cho 2 bảng chính để nhẹ bộ nhớ
    for name in ("train", "test"):
        df = tables[name]
        df["store_nbr"] = df["store_nbr"].astype("int16")
        df["family"] = df["family"].astype(str)
        df["onpromotion"] = df["onpromotion"].astype("float32")
    tables["train"]["sales"] = tables["train"]["sales"].astype("float32")
    return tables
