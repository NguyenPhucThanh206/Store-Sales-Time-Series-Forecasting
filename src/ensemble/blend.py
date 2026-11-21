"""
T13 - Ensembling (Chiến lược Blending 2 tầng: Non-DL Trees + DL GRU)
Owner : An Đức Minh Trí (24021645) & Nguyễn Mạnh Kiên (24021541) & Doãn Duy Lợi (24021549)

Chiến lược Blending 2 tầng (Hierarchical Blending) theo Mục 2.3 đề tài:
  • Tầng 1: Kết hợp bộ 3 cây Gradient Boosting (LightGBM, CatBoost, XGBoost)
            để triệt tiêu sai số cấu trúc cây riêng lẻ -> z_Tree.
  • Tầng 2: Kết hợp giữa Nhánh Deep Learning (GRU - mô hình khái quát hóa tốt nhất trên Test)
            và Nhánh Non-DL (z_Tree) với GRU giữ vai trò chủ lực (alpha = 0.55 - 0.60)
            để tránh hiện tượng Validation Overfit của các mô hình cây.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PROC_DIR  = ROOT / "data" / "processed"
PRED_VAL  = ROOT / "predictions" / "val"
PRED_TEST = ROOT / "predictions" / "test"
SUB_DIR   = ROOT / "submissions"
SUB_DIR.mkdir(parents=True, exist_ok=True)


def calc_rmsle(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def ensure_gru_validation_array(y_true_len: int = 27664) -> np.ndarray:
    val_path = PRED_VAL / "val_pred_gru.npy"
    if val_path.exists():
        arr = np.load(val_path)
        if arr.size == y_true_len:
            return arr.reshape(-1)

    import torch
    from src.dl_gru.dataset import build_dataloaders
    from src.dl_gru.model_train import GlobalMultiHorizonGRU, load_checkpoint, predict, apply_postprocessing

    _, val_loader, _, cat_maps, dynamic_cols = build_dataloaders()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GlobalMultiHorizonGRU(
        n_dynamic=len(dynamic_cols), horizon=16,
        n_stores=len(cat_maps["store_nbr"]),
        n_families=len(cat_maps["family"]),
        n_clusters=len(cat_maps["cluster"])
    )
    model = load_checkpoint(model, device=device)
    val_preds = predict(model, val_loader, device)
    val_preds = apply_postprocessing(val_preds, PROC_DIR / "zero_series.csv")
    val_preds_flat = val_preds.reshape(-1)
    np.save(val_path, val_preds_flat)
    return val_preds_flat


def run_hierarchical_blending(alpha_gru: float = 0.60):
    print("=" * 75)
    print(f"  T13: BLENDING 2 TẦNG (GRU CHỦ LỰC {alpha_gru*100:.0f}% + NHÁNH CÂY {(1-alpha_gru)*100:.0f}%)")
    print("=" * 75)

    # 1. Đọc nhãn Validation
    val_df = pd.read_parquet(PROC_DIR / "train_clean.parquet")
    val_df = val_df[val_df["is_val"] == 1].sort_values(["series_id", "date"]).reset_index(drop=True)
    y_val = val_df["sales_log1p"].to_numpy(dtype="float64")
    N_val = len(y_val)

    # 2. Nạp dự đoán Validation
    gru_val = ensure_gru_validation_array(N_val)
    xgb_val_path = PRED_VAL / "val_pred_xgboost.npy" if (PRED_VAL / "val_pred_xgboost.npy").exists() else PRED_VAL / "xgb_val.npy"

    lgb_val = np.load(PRED_VAL / "val_pred_lgbm.npy").reshape(-1)
    cat_val = np.load(PRED_VAL / "val_pred_catboost.npy").reshape(-1)
    xgb_val = np.load(xgb_val_path).reshape(-1)

    # ── TẦNG 1: Trộn nhánh cây Non-DL ──
    # Trọng số nhánh cây: lấy trung bình cộng đều giữa 3 cấu trúc cây khác nhau
    tree_val = (lgb_val + cat_val + xgb_val) / 3.0

    print("── 1. ĐIỂM RMSLE KIỂM ĐỊNH TỪNG PHÂN HỆ ──")
    print(f"  • LightGBM       : {calc_rmsle(y_val, lgb_val):.5f}")
    print(f"  • CatBoost       : {calc_rmsle(y_val, cat_val):.5f}")
    print(f"  • XGBoost        : {calc_rmsle(y_val, xgb_val):.5f}")
    print(f"  • Tầng 1 (3 Cây) : {calc_rmsle(y_val, tree_val):.5f}")
    print(f"  • GRU            : {calc_rmsle(y_val, gru_val):.5f}")

    # ── TẦNG 2: Hòa trộn GRU (chủ lực) + Nhánh Cây ──
    val_blend_final = alpha_gru * gru_val + (1.0 - alpha_gru) * tree_val
    print(f"\n── 2. KẾT HỢP TẦNG 2: {alpha_gru*100:.0f}% GRU + {(1-alpha_gru)*100:.0f}% Nhánh Cây ──")
    print(f"  >>> RMSLE Validation của tổ hợp 2 tầng: {calc_rmsle(y_val, val_blend_final):.5f}\n")

    np.save(PRED_VAL / "val_pred_blend.npy", val_blend_final.astype("float32"))

    # ── 3. Áp dụng vào tập Test ──
    xgb_test_path = PRED_TEST / "test_pred_xgboost.npy" if (PRED_TEST / "test_pred_xgboost.npy").exists() else PRED_TEST / "xgb_test.npy"

    lgb_test = np.load(PRED_TEST / "test_pred_lgbm.npy").reshape(-1)
    cat_test = np.load(PRED_TEST / "test_pred_catboost.npy").reshape(-1)
    xgb_test = np.load(xgb_test_path).reshape(-1)
    gru_test = np.load(PRED_TEST / "test_pred_gru.npy").reshape(-1)

    # Tầng 1 Test
    tree_test = (lgb_test + cat_test + xgb_test) / 3.0
    # Tầng 2 Test
    test_blend_final = alpha_gru * gru_test + (1.0 - alpha_gru) * tree_test

    np.save(PRED_TEST / "test_pred_blend.npy", test_blend_final.astype("float32"))
    print(f"  ✓ Đã lưu mảng test dự báo Blending: {PRED_TEST / 'test_pred_blend.npy'}")

    # ── 4. Hậu xử lý và xuất file submission_final.csv ──
    from src.postprocess.make_submission import run as make_sub
    sub_path = SUB_DIR / "submission_final.csv"
    make_sub(
        pred_path=PRED_TEST / "test_pred_blend.npy",
        out_path=sub_path,
        zero_tail=True,
    )

    print("\n" + "=" * 75)
    print(f"  HOÀN TẤT! Tệp nộp mới đã sẵn sàng: {sub_path}")
    print("=" * 75)


if __name__ == "__main__":
    run_hierarchical_blending(alpha_gru=0.60)
