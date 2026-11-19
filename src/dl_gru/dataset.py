"""
Owner  : Doãn Duy Lợi (24021549) – Task T04
Mô tả  : Pipeline cửa sổ trượt L=30 ngày quá khứ → 16 ngày tương lai
         cho 1.782 chuỗi (54 store × 33 family).
         Bao gồm:
           • StoreSalesDataset  – PyTorch Dataset cho Training / Validation
           • StoreSalesTestDataset – PyTorch Dataset cho Inference trên test.csv
           • build_dataloaders()  – Hàm tiện ích tạo DataLoader train/val/test
Input  : data/processed/train_clean.parquet  (output của Thành – T02)
         data/raw/stores.csv                 (thông tin cluster)
         data/processed/val_dates.json       (danh sách ngày Validation)
         data/raw/test.csv                   (tập test Kaggle)
"""

import json
import pathlib
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

# ──────────────────────────────────────────────────────────────
# 0.  ĐƯỜNG DẪN MẶC ĐỊNH
# ──────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[2]        # Store-Sales-…/
RAW  = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

# ──────────────────────────────────────────────────────────────
# 1.  BẢNG MÃ HÓA CÁC BIẾN PHÂN LOẠI (categorical encoding)
# ──────────────────────────────────────────────────────────────
def _build_category_maps(
    train_df: pd.DataFrame,
    stores_df: pd.DataFrame,
) -> Dict[str, Dict]:
    """
    Xây dựng bảng ánh xạ category → index (bắt đầu từ 0) cho 3 biến:
      • store_nbr  : 1‑54  → 0‑53
      • family     : 33 loại → 0‑32
      • cluster    : giá trị trong stores.csv → 0‑(K−1)

    Trả về dict: { "store_nbr": {val: idx}, "family": {...}, "cluster": {...} }
    """
    cat_maps: Dict[str, Dict] = {}

    # store_nbr
    unique_stores = sorted(train_df["store_nbr"].unique())
    cat_maps["store_nbr"] = {v: i for i, v in enumerate(unique_stores)}

    # family
    unique_families = sorted(train_df["family"].unique())
    cat_maps["family"] = {v: i for i, v in enumerate(unique_families)}

    # cluster (lấy từ stores.csv, merge qua store_nbr)
    unique_clusters = sorted(stores_df["cluster"].unique())
    cat_maps["cluster"] = {v: i for i, v in enumerate(unique_clusters)}

    return cat_maps


def _store_to_cluster(stores_df: pd.DataFrame) -> Dict[int, int]:
    """Trả về dict store_nbr → cluster."""
    return dict(zip(stores_df["store_nbr"], stores_df["cluster"]))


# ──────────────────────────────────────────────────────────────
# 2.  DYNAMIC FEATURES (chuỗi đặc trưng thay đổi theo ngày)
# ──────────────────────────────────────────────────────────────
# Các cột được chọn làm chuỗi đặc trưng đầu vào (dynamic features).
# Nếu cột nào không tồn tại trong parquet thì bỏ qua.
_DYNAMIC_COLS_CANDIDATES = [
    "log_sales",        # biến mục tiêu (đã log1p)
    "onpromotion",      # số sản phẩm khuyến mãi
    "sin_dow",          # chu kỳ ngày trong tuần (sin)
    "cos_dow",          # chu kỳ ngày trong tuần (cos)
    "sin_month",        # chu kỳ tháng trong năm (sin)
    "cos_month",        # chu kỳ tháng trong năm (cos)
    "sin_day",          # chu kỳ ngày trong tháng (sin)
    "cos_day",          # chu kỳ ngày trong tháng (cos)
    "dayofweek",        # 0=Mon … 6=Sun
    "day",              # ngày trong tháng
    "month",            # tháng
    "is_payday",        # cờ ngày lương (15 & cuối tháng)
    "is_holiday",       # cờ ngày lễ
    "days_to_holiday",  # khoảng cách đến ngày lễ gần nhất
    "oil_price",        # giá dầu (đã nội suy)
    "oil_ma7",          # trung bình trượt giá dầu 7 ngày
    "oil_ma14",         # trung bình trượt giá dầu 14 ngày
    "transactions",     # số giao dịch (nếu có)
]


def _add_cyclic_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo các đặc trưng chu kỳ thời gian (sin/cos) cho mạng GRU."""
    df = df.copy()
    if "date" in df.columns:
        dt = pd.to_datetime(df["date"])
        dow   = dt.dt.dayofweek
        day   = dt.dt.day
        month = dt.dt.month

        df["dayofweek"] = dow.astype(np.float32)
        df["day"]       = day.astype(np.float32)
        df["month"]     = month.astype(np.float32)

        df["sin_dow"]   = np.sin(2 * np.pi * dow / 7.0).astype(np.float32)
        df["cos_dow"]   = np.cos(2 * np.pi * dow / 7.0).astype(np.float32)
        df["sin_month"] = np.sin(2 * np.pi * (month - 1) / 12.0).astype(np.float32)
        df["cos_month"] = np.cos(2 * np.pi * (month - 1) / 12.0).astype(np.float32)
        df["sin_day"]   = np.sin(2 * np.pi * (day - 1) / 31.0).astype(np.float32)
        df["cos_day"]   = np.cos(2 * np.pi * (day - 1) / 31.0).astype(np.float32)
    return df


def _select_dynamic_cols(df: pd.DataFrame) -> List[str]:
    """Trả về danh sách cột dynamic thực sự có mặt trong DataFrame."""
    return [c for c in _DYNAMIC_COLS_CANDIDATES if c in df.columns]


# ──────────────────────────────────────────────────────────────
# 3.  XÂY DỰNG CHUỖI SLIDING WINDOW CHO TỪNG CẶP (store, family)
# ──────────────────────────────────────────────────────────────
def _prepare_series(
    df: pd.DataFrame,
    stores_df: pd.DataFrame,
    cat_maps: Dict[str, Dict],
    lookback: int = 30,
    horizon: int = 16,
    mode: str = "train",          # "train" | "val" | "test"
    val_dates: Optional[List[str]] = None,
) -> List[Dict[str, np.ndarray]]:
    """
    Tạo danh sách các mẫu (sample) từ DataFrame.

    Mỗi mẫu gồm:
      • x_dynamic  : (lookback, n_dynamic_features)   – chuỗi quá khứ
      • y_target   : (horizon,)                        – chuỗi tương lai (log_sales)
      • store_idx  : int   – chỉ số embedding store
      • family_idx : int   – chỉ số embedding family
      • cluster_idx: int   – chỉ số embedding cluster

    Quy tắc:
      - mode="train" : lấy TẤT CẢ cửa sổ trượt hợp lệ, loại trừ ngày val.
      - mode="val"   : chỉ lấy 1 cửa sổ cuối sao cho target = val_dates.
      - mode="test"  : lấy 1 cửa sổ cuối cùng (không có target, y_target=0).
    """
    store_cluster = _store_to_cluster(stores_df)
    dynamic_cols  = _select_dynamic_cols(df)

    # Đảm bảo cột date là datetime & sắp xếp
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)

    samples: List[Dict[str, np.ndarray]] = []

    grouped = df.groupby(["store_nbr", "family"], sort=True)

    for (store, family), grp in grouped:
        grp = grp.sort_values("date").reset_index(drop=True)

        # --- Mã hóa categorical ---
        s_idx = cat_maps["store_nbr"][store]
        f_idx = cat_maps["family"][family]
        c_idx = cat_maps["cluster"][store_cluster[store]]

        # --- Ma trận dynamic ---
        dyn = grp[dynamic_cols].values.astype(np.float32)   # (T, D)
        dates = grp["date"].values                            # numpy datetime64

        T = len(grp)

        if mode == "test":
            # Chỉ lấy lookback ngày cuối cùng (không có target)
            if T < lookback:
                # Pad nếu chuỗi quá ngắn
                pad_len = lookback - T
                dyn_pad = np.zeros((pad_len, dyn.shape[1]), dtype=np.float32)
                x_dyn   = np.concatenate([dyn_pad, dyn], axis=0)
            else:
                x_dyn = dyn[-lookback:]
            samples.append({
                "x_dynamic"  : x_dyn,
                "y_target"   : np.zeros(horizon, dtype=np.float32),
                "store_idx"  : s_idx,
                "family_idx" : f_idx,
                "cluster_idx": c_idx,
            })
            continue

        if mode == "val":
            # Lấy 1 cửa sổ duy nhất sao cho phần target bắt đầu ở val_dates[0]
            if val_dates is None:
                raise ValueError("val_dates phải được cung cấp cho mode='val'")
            val_start = pd.Timestamp(val_dates[0])
            # Tìm vị trí ngày bắt đầu validation trong chuỗi
            date_series = pd.Series(dates)
            idx_val = date_series[date_series == val_start].index
            if len(idx_val) == 0:
                continue  # chuỗi này không có ngày val
            t_val = idx_val[0]
            if t_val < lookback:
                continue  # không đủ lookback
            x_dyn    = dyn[t_val - lookback : t_val]
            y_target = dyn[t_val : t_val + horizon, 0]  # cột 0 = log_sales
            if len(y_target) < horizon:
                continue
            samples.append({
                "x_dynamic"  : x_dyn,
                "y_target"   : y_target,
                "store_idx"  : s_idx,
                "family_idx" : f_idx,
                "cluster_idx": c_idx,
            })
            continue

        # mode == "train" ─ sliding window
        if val_dates is not None:
            val_start = pd.Timestamp(val_dates[0])
        else:
            val_start = None

        for t in range(lookback, T - horizon + 1):
            # Loại trừ cửa sổ chồng lên val
            if val_start is not None and pd.Timestamp(dates[t]) >= val_start:
                break

            x_dyn    = dyn[t - lookback : t]
            y_target = dyn[t : t + horizon, 0]        # log_sales
            samples.append({
                "x_dynamic"  : x_dyn,
                "y_target"   : y_target,
                "store_idx"  : s_idx,
                "family_idx" : f_idx,
                "cluster_idx": c_idx,
            })

    return samples


# ──────────────────────────────────────────────────────────────
# 4.  PYTORCH DATASET WRAPPERS
# ──────────────────────────────────────────────────────────────
class StoreSalesDataset(Dataset):
    """
    PyTorch Dataset cho training / validation.

    Trả về tuple gồm:
      • x_dynamic   : Tensor (lookback, D)
      • cat_ids     : Tensor (3,)   = [store_idx, family_idx, cluster_idx]
      • y_target    : Tensor (horizon,)
    """

    def __init__(self, samples: List[Dict[str, np.ndarray]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        x_dyn   = torch.from_numpy(s["x_dynamic"])                       # (L, D)
        cat_ids = torch.tensor(
            [s["store_idx"], s["family_idx"], s["cluster_idx"]],
            dtype=torch.long,
        )                                                                  # (3,)
        y       = torch.from_numpy(s["y_target"])                         # (H,)
        return x_dyn, cat_ids, y


class StoreSalesTestDataset(Dataset):
    """
    PyTorch Dataset cho Inference (test.csv) – y_target luôn = 0.
    """

    def __init__(self, samples: List[Dict[str, np.ndarray]]):
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        s = self.samples[idx]
        x_dyn   = torch.from_numpy(s["x_dynamic"])
        cat_ids = torch.tensor(
            [s["store_idx"], s["family_idx"], s["cluster_idx"]],
            dtype=torch.long,
        )
        return x_dyn, cat_ids


# ──────────────────────────────────────────────────────────────
# 5.  HÀM TIỆN ÍCH: build_dataloaders()
# ──────────────────────────────────────────────────────────────
def build_dataloaders(
    lookback: int = 30,
    horizon: int = 16,
    batch_size: int = 1024,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dict[str, Dict], List[str]]:
    """
    Đọc dữ liệu đã xử lý → xây dựng & trả về 3 DataLoader (train, val, test)
    cùng bảng mã hoá categorical & danh sách dynamic columns.

    Returns
    -------
    train_loader, val_loader, test_loader, cat_maps, dynamic_cols
    """

    # --- Đọc dữ liệu ---
    train_df  = pd.read_parquet(PROC / "train_clean.parquet")
    stores_df = pd.read_csv(RAW / "stores.csv")
    test_df   = pd.read_csv(RAW / "test.csv")

    train_df["date"] = pd.to_datetime(train_df["date"])
    train_df["family"] = train_df["family"].astype(str)
    test_df["family"]  = test_df["family"].astype(str)

    # Chuẩn hoá cột log_sales và các cột thời gian cơ bản nếu chưa có
    if "log_sales" not in train_df.columns:
        if "sales_log1p" in train_df.columns:
            train_df["log_sales"] = train_df["sales_log1p"].fillna(0.0).astype(np.float32)
        elif "sales" in train_df.columns:
            train_df["log_sales"] = np.log1p(train_df["sales"].clip(lower=0)).fillna(0.0).astype(np.float32)
    else:
        train_df["log_sales"] = train_df["log_sales"].fillna(0.0).astype(np.float32)

    # Bổ sung các đặc trưng chu kỳ thời gian (sin/cos encoding)
    train_df = _add_cyclic_features(train_df)

    val_json_path = PROC / "val_dates.json"
    if val_json_path.exists():
        with open(val_json_path, "r") as f:
            val_dates: List[str] = json.load(f)
    elif "is_val" in train_df.columns:
        val_dates = sorted(train_df.loc[train_df["is_val"] == 1, "date"].dt.strftime("%Y-%m-%d").unique().tolist())
    else:
        all_d = sorted(train_df["date"].dt.strftime("%Y-%m-%d").unique().tolist())
        val_dates = all_d[-horizon:]

    # --- Bảng mã hoá ---
    cat_maps = _build_category_maps(train_df, stores_df)

    # --- Bổ sung các cột dynamic tối thiểu cho test_df nếu thiếu ---
    test_df["date"] = pd.to_datetime(test_df["date"])
    if "log_sales" not in test_df.columns:
        test_df["log_sales"] = 0.0

    # Bổ sung các đặc trưng chu kỳ thời gian (sin/cos encoding) cho test
    test_df = _add_cyclic_features(test_df)

    dynamic_cols = _select_dynamic_cols(train_df)

    # ---- Ghép test_df vào cuối train_df để tạo cửa sổ lookback cho test ----
    # Lấy lookback ngày cuối của mỗi chuỗi từ train_df, nối với test_df
    train_df["date"] = pd.to_datetime(train_df["date"])
    tail_parts = []
    for (store, family), grp in train_df.groupby(["store_nbr", "family"]):
        grp_sorted = grp.sort_values("date")
        tail_parts.append(grp_sorted.tail(lookback))
    tail_df = pd.concat(tail_parts, ignore_index=True)

    # Đảm bảo test_df có các cột cần thiết (fill 0 nếu thiếu)
    for col in dynamic_cols:
        if col not in test_df.columns:
            test_df[col] = 0.0

    combined_test = pd.concat(
        [tail_df[["date", "store_nbr", "family"] + dynamic_cols],
         test_df[["date", "store_nbr", "family"] + dynamic_cols]],
        ignore_index=True,
    )

    # --- Tạo samples ---
    train_samples = _prepare_series(
        train_df, stores_df, cat_maps,
        lookback=lookback, horizon=horizon,
        mode="train", val_dates=val_dates,
    )
    val_samples = _prepare_series(
        train_df, stores_df, cat_maps,
        lookback=lookback, horizon=horizon,
        mode="val", val_dates=val_dates,
    )
    test_samples = _prepare_series(
        combined_test, stores_df, cat_maps,
        lookback=lookback, horizon=horizon,
        mode="test",
    )

    # --- DataLoaders ---
    use_pin = torch.cuda.is_available()
    train_ds = StoreSalesDataset(train_samples)
    val_ds   = StoreSalesDataset(val_samples)
    test_ds  = StoreSalesTestDataset(test_samples)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=use_pin, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_pin,
    )

    print(f"[Dataset] Train samples : {len(train_ds):,}")
    print(f"[Dataset] Val   samples : {len(val_ds):,}")
    print(f"[Dataset] Test  samples : {len(test_ds):,}")
    print(f"[Dataset] Dynamic cols  : {dynamic_cols}")
    print(f"[Dataset] Cat embed sizes: "
          f"store={len(cat_maps['store_nbr'])}, "
          f"family={len(cat_maps['family'])}, "
          f"cluster={len(cat_maps['cluster'])}")

    return train_loader, val_loader, test_loader, cat_maps, dynamic_cols


# ──────────────────────────────────────────────────────────────
# 6.  ENTRYPOINT (chạy thử độc lập)
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    tl, vl, ttl, cm, dc = build_dataloaders(batch_size=512)
    # In 1 batch đầu tiên để kiểm tra shape
    for x_dyn, cat_ids, y in tl:
        print(f"  x_dynamic shape : {x_dyn.shape}")   # (B, 30, D)
        print(f"  cat_ids shape   : {cat_ids.shape}")  # (B, 3)
        print(f"  y_target shape  : {y.shape}")        # (B, 16)
        break
