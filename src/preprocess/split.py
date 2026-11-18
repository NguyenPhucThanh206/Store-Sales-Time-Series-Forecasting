"""Biến đổi log1p và tách N ngày cuối làm Validation."""
import numpy as np
import pandas as pd


def add_log1p(df: pd.DataFrame) -> pd.DataFrame:
    """sales_log1p = log1p(sales). Giữ nguyên cột sales gốc. NaN (ngày đóng cửa) giữ NaN."""
    if (df["sales"] < 0).any():
        raise ValueError("Có sales âm, cần kiểm tra lại dữ liệu")
    df["sales_log1p"] = np.log1p(df["sales"]).astype("float32")
    return df


def mark_validation(df: pd.DataFrame, val_days: int = 16) -> pd.DataFrame:
    """is_val = 1 cho val_days ngày cuối của train (vd 2017-07-31 -> 2017-08-15)."""
    last = df["date"].max()
    start = last - pd.Timedelta(days=val_days - 1)
    df["is_val"] = (df["date"] >= start).astype("int8")
    n = df.loc[df["is_val"] == 1, "date"].nunique()
    if n != val_days:
        raise AssertionError(f"Validation có {n} ngày, kỳ vọng {val_days}")
    return df
