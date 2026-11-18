"""Làm sạch: lưới ngày đầy đủ, ngày đóng cửa, cắt chuỗi 0 đầu, chuỗi toàn 0."""
import numpy as np
import pandas as pd


def build_series_map(test: pd.DataFrame) -> pd.DataFrame:
    """series_id cố định 0..N-1, sắp theo (store_nbr, family); lấy từ test (đủ 1.782 chuỗi)."""
    m = (test[["store_nbr", "family"]].drop_duplicates()
         .sort_values(["store_nbr", "family"]).reset_index(drop=True))
    m["series_id"] = np.arange(len(m), dtype="int16")
    return m


def flag_closed_days(dates: pd.Series) -> pd.Series:
    """Ngày 1/1 và 25/12: cửa hàng đóng cửa / không có dữ liệu."""
    return ((dates.dt.month == 1) & (dates.dt.day == 1)) | \
           ((dates.dt.month == 12) & (dates.dt.day == 25))


def build_full_grid(train: pd.DataFrame, series_map: pd.DataFrame) -> pd.DataFrame:
    """Lưới đầy đủ (ngày x chuỗi). Train gốc thiếu ngày 25/12 hằng năm -> thêm lại."""
    pairs_train = set(map(tuple, train[["store_nbr", "family"]].drop_duplicates().to_numpy()))
    pairs_map = set(map(tuple, series_map[["store_nbr", "family"]].to_numpy()))
    if not pairs_train <= pairs_map:
        raise ValueError("Train có chuỗi (store, family) không xuất hiện trong test")

    dates = pd.date_range(train["date"].min(), train["date"].max(), freq="D")
    idx = pd.MultiIndex.from_product([series_map["series_id"], dates], names=["series_id", "date"])
    grid = idx.to_frame(index=False).merge(series_map, on="series_id", how="left")
    grid = grid.merge(train[["date", "store_nbr", "family", "sales", "onpromotion"]],
                      on=["date", "store_nbr", "family"], how="left")
    grid["onpromotion"] = grid["onpromotion"].fillna(0).astype("float32")
    return grid


def clean_train(train: pd.DataFrame, test: pd.DataFrame, zero_tail_days: int = 28):
    """Trả về (train_clean, test_clean, zero_info)."""
    series_map = build_series_map(test)
    grid = build_full_grid(train, series_map)

    # 1) Ngày đóng cửa -> sales = NaN (giữ dòng để lag/rolling không lệch lịch)
    grid["is_closed"] = flag_closed_days(grid["date"]).astype("int8")
    grid.loc[grid["is_closed"] == 1, "sales"] = np.nan

    # 2) Cắt phần đầu toàn 0 của từng chuỗi (cửa hàng chưa mở / chưa bán sản phẩm)
    first_sale = grid.loc[grid["sales"] > 0].groupby("series_id")["date"].min()
    all_zero_ids = sorted(set(series_map["series_id"]) - set(first_sale.index))
    grid["first_sale"] = grid["series_id"].map(first_sale)
    grid = grid[grid["date"] >= grid["first_sale"]]  # NaT -> False: chuỗi toàn 0 bị loại
    grid = grid.drop(columns="first_sale")

    # 3) Chuỗi toàn 0 (loại khỏi train) + chuỗi 0 liên tục ở đuôi (giữ lại, chỉ ghi nhận)
    last_date = grid["date"].max()
    tail = grid[grid["date"] > last_date - pd.Timedelta(days=zero_tail_days)]
    tail_sum = tail.groupby("series_id")["sales"].sum(min_count=1)
    zero_tail_ids = sorted(tail_sum[tail_sum == 0].index)

    zero_info = pd.concat([
        series_map[series_map["series_id"].isin(all_zero_ids)].assign(kind="all_zero"),
        series_map[series_map["series_id"].isin(zero_tail_ids)].assign(kind=f"zero_last_{zero_tail_days}d"),
    ], ignore_index=True)

    grid = grid.sort_values(["series_id", "date"]).reset_index(drop=True)
    grid["family"] = grid["family"].astype("category")

    test_clean = test.merge(series_map, on=["store_nbr", "family"], how="left")
    test_clean["family"] = test_clean["family"].astype("category")
    test_clean = test_clean.sort_values(["series_id", "date"]).reset_index(drop=True)
    return grid, test_clean, zero_info, series_map
