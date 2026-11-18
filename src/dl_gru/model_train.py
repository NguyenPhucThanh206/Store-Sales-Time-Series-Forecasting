"""
Owner  : Doãn Duy Lợi (24021549) – Task T09 (Tuần 1) & T12 (Tuần 2)
Mô tả  : Kiến trúc Global Multi-Horizon GRU kèm Embedding (store_nbr, family, cluster)
         và toàn bộ vòng lặp huấn luyện trên GPU.

Kiến trúc mạng:
  ┌──────────────────────────────────────────────────────────────┐
  │  Input chuỗi dynamic   (B, L=30, D)                        │
  │  Input categorical ids  (B, 3)  →  3 × Embedding layers    │
  │                                                              │
  │  GRU (2 layers, hidden=128, dropout=0.2)                    │
  │      ↓                                                       │
  │  Lấy hidden state cuối  (B, hidden_dim)                     │
  │      ↓                                                       │
  │  Concat với Embedding vector (B, hidden_dim + emb_total)    │
  │      ↓                                                       │
  │  FC → ReLU → Dropout → FC → output (B, H=16)               │
  └──────────────────────────────────────────────────────────────┘

Output : predictions/val/val_pred_gru.npy   (trên thang log1p)
         predictions/test/test_pred_gru.npy (trên thang log1p)
"""

import pathlib
import time
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")  # Chạy headless, không mở cửa sổ GUI
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

# ──────────────────────────────────────────────────────────────
# 0.  ĐƯỜNG DẪN MẶC ĐỊNH
# ──────────────────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).resolve().parents[2]
PROC      = ROOT / "data" / "processed"
PRED_VAL  = ROOT / "predictions" / "val"
PRED_TEST = ROOT / "predictions" / "test"
FIGURES   = ROOT / "reports" / "figures"
PRED_VAL.mkdir(parents=True, exist_ok=True)
PRED_TEST.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)


# ══════════════════════════════════════════════════════════════
# 1.  KIẾN TRÚC MÔ HÌNH GRU
# ══════════════════════════════════════════════════════════════
class GlobalMultiHorizonGRU(nn.Module):
    """
    Global Multi-Horizon GRU cho bài toán dự báo chuỗi thời gian.

    Mô hình học toàn cục (global) trên toàn bộ 1.782 chuỗi, sử dụng
    các lớp Embedding để biểu diễn thông tin tĩnh (store, family, cluster)
    và GRU để nắm bắt động lực chuỗi thời gian.

    Xuất trực tiếp vector dự báo H=16 ngày (direct multi-step forecasting)
    để tránh sai số tích lũy (error accumulation) so với dự báo đệ quy.

    Parameters
    ----------
    n_dynamic     : Số lượng đặc trưng dynamic trong chuỗi đầu vào.
    hidden_dim    : Kích thước lớp ẩn của GRU.
    n_gru_layers  : Số lớp GRU chồng nhau.
    horizon       : Số bước dự báo tương lai (H=16).
    n_stores      : Số cửa hàng (cho Embedding).
    n_families    : Số nhóm mặt hàng (cho Embedding).
    n_clusters    : Số cluster (cho Embedding).
    emb_dim_store : Chiều Embedding cho store.
    emb_dim_family: Chiều Embedding cho family.
    emb_dim_cluster: Chiều Embedding cho cluster.
    dropout       : Tỷ lệ Dropout.
    """

    def __init__(
        self,
        n_dynamic: int,
        hidden_dim: int = 128,
        n_gru_layers: int = 2,
        horizon: int = 16,
        n_stores: int = 54,
        n_families: int = 33,
        n_clusters: int = 17,
        emb_dim_store: int = 8,
        emb_dim_family: int = 8,
        emb_dim_cluster: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()

        # ----- Embedding layers cho biến tĩnh -----
        self.emb_store   = nn.Embedding(n_stores,   emb_dim_store)
        self.emb_family  = nn.Embedding(n_families,  emb_dim_family)
        self.emb_cluster = nn.Embedding(n_clusters,  emb_dim_cluster)
        emb_total = emb_dim_store + emb_dim_family + emb_dim_cluster  # 20

        # ----- GRU encoder -----
        self.gru = nn.GRU(
            input_size=n_dynamic,
            hidden_size=hidden_dim,
            num_layers=n_gru_layers,
            batch_first=True,
            dropout=dropout if n_gru_layers > 1 else 0.0,
        )
        self.norm = nn.LayerNorm(hidden_dim)

        # ----- Fully-Connected decoder head (với LayerNorm & Residual) -----
        fc_input = hidden_dim + emb_total
        self.fc = nn.Sequential(
            nn.Linear(fc_input, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, horizon),
        )

        self.horizon = horizon

    def forward(
        self,
        x_dynamic: torch.Tensor,   # (B, L, D)
        cat_ids: torch.Tensor,     # (B, 3)   [store_idx, family_idx, cluster_idx]
    ) -> torch.Tensor:
        """
        Forward pass với Residual Skip Connection.

        Returns
        -------
        output : Tensor (B, H)  –  dự báo H=16 bước trên thang log1p.
        """
        # ----- GRU -----
        gru_out, _ = self.gru(x_dynamic)              # (B, L, hidden_dim)
        h_last = self.norm(gru_out[:, -1, :])         # (B, hidden_dim)

        # ----- Embedding -----
        e_store   = self.emb_store(cat_ids[:, 0])    # (B, emb_dim_store)
        e_family  = self.emb_family(cat_ids[:, 1])   # (B, emb_dim_family)
        e_cluster = self.emb_cluster(cat_ids[:, 2])  # (B, emb_dim_cluster)

        emb_cat = torch.cat([e_store, e_family, e_cluster], dim=1)  # (B, emb_total)

        # ----- Concat & FC Head -----
        combined = torch.cat([h_last, emb_cat], dim=1)              # (B, hidden+emb)
        delta    = self.fc(combined)                                 # (B, H)

        # ----- Residual Skip Connection -----
        # Cột 0 của x_dynamic chính là log_sales.
        # Lấy mức trung bình quá khứ làm nền tảng, mạng chỉ cần học độ lệch (delta)
        base_level = x_dynamic[:, :, 0].mean(dim=1, keepdim=True)    # (B, 1)
        output = delta + base_level                                   # (B, H)

        return output


# ══════════════════════════════════════════════════════════════
# 2.  HÀM TÍNH RMSLE TRÊN THANG LOG1P
# ══════════════════════════════════════════════════════════════
def rmsle_on_log(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """
    Tính RMSLE khi pred và target đều đã ở thang log(1+y).

    RMSLE = sqrt( mean( (log(1+ŷ) − log(1+y))² ) )

    Vì dữ liệu đã được biến đổi log1p trước → RMSLE = RMSE trên thang log.
    """
    return torch.sqrt(nn.functional.mse_loss(pred, target))


# ══════════════════════════════════════════════════════════════
# 3.  VÒNG LẶP HUẤN LUYỆN
# ══════════════════════════════════════════════════════════════
def train_model(
    model: GlobalMultiHorizonGRU,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_epochs: int = 60,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 8,
    verbose: bool = True,
) -> Dict[str, List[float]]:
    """
    Huấn luyện mô hình GRU với EarlyStopping và ReduceLROnPlateau.

    Parameters
    ----------
    model         : Instance của GlobalMultiHorizonGRU.
    train_loader  : DataLoader cho tập huấn luyện.
    val_loader    : DataLoader cho tập kiểm định.
    device        : torch.device (cuda / cpu).
    n_epochs      : Số epoch tối đa.
    lr            : Learning rate ban đầu.
    weight_decay  : Hệ số L2 regularization.
    patience      : Số epoch cho phép val loss không giảm (EarlyStopping).
    verbose       : In tiến trình huấn luyện.

    Returns
    -------
    history : Dict với 2 key: "train_loss" và "val_loss" (danh sách RMSLE mỗi epoch).
    """
    model = model.to(device)

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3,
    )
    criterion = nn.MSELoss()   # RMSLE = sqrt(MSE) trên thang log

    history: Dict[str, List[float]] = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state    = None
    epochs_no_improve = 0

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        # ── Train ──
        model.train()
        train_mse_sum = 0.0
        train_count   = 0

        for x_dyn, cat_ids, y in train_loader:
            x_dyn   = x_dyn.to(device)
            cat_ids = cat_ids.to(device)
            y       = y.to(device)

            pred = model(x_dyn, cat_ids)       # (B, H)
            loss = criterion(pred, y)

            optimizer.zero_grad()
            loss.backward()
            # Gradient clipping để tránh exploding gradients
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()

            train_mse_sum += loss.item() * y.size(0)
            train_count   += y.size(0)

        train_rmsle = np.sqrt(train_mse_sum / train_count)

        # ── Validation ──
        model.eval()
        val_mse_sum = 0.0
        val_count   = 0

        with torch.no_grad():
            for x_dyn, cat_ids, y in val_loader:
                x_dyn   = x_dyn.to(device)
                cat_ids = cat_ids.to(device)
                y       = y.to(device)

                pred = model(x_dyn, cat_ids)
                loss = criterion(pred, y)

                val_mse_sum += loss.item() * y.size(0)
                val_count   += y.size(0)

        val_rmsle = np.sqrt(val_mse_sum / val_count)

        # ── Ghi nhận ──
        history["train_loss"].append(train_rmsle)
        history["val_loss"].append(val_rmsle)

        scheduler.step(val_rmsle)

        elapsed = time.time() - t0

        if verbose:
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"Epoch {epoch:3d}/{n_epochs} │ "
                f"Train RMSLE: {train_rmsle:.5f} │ "
                f"Val RMSLE: {val_rmsle:.5f} │ "
                f"LR: {current_lr:.2e} │ "
                f"Time: {elapsed:.1f}s"
            )

        # ── Early Stopping ──
        if val_rmsle < best_val_loss:
            best_val_loss     = val_rmsle
            best_state        = model.state_dict().copy()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            if verbose:
                print(f"\n[EarlyStopping] Dừng tại epoch {epoch}. "
                      f"Best Val RMSLE = {best_val_loss:.5f}")
            break

    # Khôi phục trọng số tốt nhất
    if best_state is not None:
        model.load_state_dict(best_state)

    if verbose:
        print(f"\n✓ Huấn luyện hoàn tất. Best Val RMSLE = {best_val_loss:.5f}")

    return history


# ══════════════════════════════════════════════════════════════
# 4.  HÀM SINH DỰ BÁO (Inference)
# ══════════════════════════════════════════════════════════════
@torch.no_grad()
def predict(
    model: GlobalMultiHorizonGRU,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    """
    Sinh dự báo cho toàn bộ DataLoader.

    Returns
    -------
    predictions : ndarray (N, H) trên thang log(1+sales).
    """
    model.eval()
    all_preds = []

    for batch in loader:
        # Phân biệt train/val (3 tensors) vs test (2 tensors)
        if len(batch) == 3:
            x_dyn, cat_ids, _ = batch
        else:
            x_dyn, cat_ids = batch

        x_dyn   = x_dyn.to(device)
        cat_ids = cat_ids.to(device)

        pred = model(x_dyn, cat_ids)       # (B, H)
        all_preds.append(pred.cpu().numpy())

    return np.concatenate(all_preds, axis=0)


# ══════════════════════════════════════════════════════════════
# 5.  HẬU XỬ LÝ (Post-processing) & VẼ ĐỒ THỊ
# ══════════════════════════════════════════════════════════════
def apply_postprocessing(
    preds: np.ndarray,
    zero_series_path: Optional[pathlib.Path] = None,
) -> np.ndarray:
    """
    Hậu xử lý mảng dự báo:
      1. Ép tất cả giá trị âm về 0.0 (doanh số không thể âm).
      2. Nếu có zero_series.csv, ép các chuỗi toàn 0 (all_zero) về đúng 0.0.
    """
    # 1. Ép mọi giá trị âm về 0.0
    preds = np.clip(preds, a_min=0.0, a_max=None)

    # 2. Xử lý chuỗi toàn 0 nếu có file thông tin
    if zero_series_path is not None and zero_series_path.exists():
        try:
            zero_df = pd.read_csv(zero_series_path)
            if "series_id" in zero_df.columns and "kind" in zero_df.columns:
                all_zero_ids = zero_df.loc[zero_df["kind"] == "all_zero", "series_id"].unique()
                # Kiểm tra số chuỗi khớp kích thước (1782 chuỗi)
                if len(preds) == 1782:
                    valid_ids = [i for i in all_zero_ids if 0 <= i < 1782]
                    preds[valid_ids, :] = 0.0
                    print(f"  [Post-process] Đã ép {len(valid_ids)} chuỗi toàn 0 về 0.0")
        except Exception as e:
            print(f"  [Post-process] Cảnh báo khi đọc zero_series.csv: {e}")

    return preds


def plot_loss_curve(
    history: Dict[str, List[float]],
    save_path: pathlib.Path,
) -> None:
    """
    Vẽ biểu đồ quá trình học tập (Train vs Val RMSLE) qua các Epoch và lưu ảnh.
    Dùng để chèn vào Slide thuyết trình và Báo cáo nghiệm thu.
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)
    best_idx   = int(np.argmin(history["val_loss"]))
    best_epoch = best_idx + 1
    best_val   = history["val_loss"][best_idx]

    plt.figure(figsize=(9, 5), dpi=150)
    plt.plot(epochs, history["train_loss"], label="Train RMSLE", color="#1f77b4", lw=2)
    plt.plot(epochs, history["val_loss"],   label="Validation RMSLE", color="#ff7f0e", lw=2)

    # Đánh dấu Best Epoch
    plt.axvline(x=best_epoch, color="red", linestyle="--", alpha=0.7,
                label=f"Kỷ lục Epoch {best_epoch} (RMSLE = {best_val:.5f})")
    plt.scatter([best_epoch], [best_val], color="red", s=70, zorder=5)

    plt.title("Quá trình huấn luyện mạng Global Multi-Horizon GRU", fontsize=13, fontweight="bold")
    plt.xlabel("Epoch", fontsize=11)
    plt.ylabel("RMSLE (trên thang log1p)", fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"  [Plot] Biểu đồ Loss Curve đã được lưu → {save_path}")


# ══════════════════════════════════════════════════════════════
# 6.  SAVE / LOAD TRỌNG SỐ
# ══════════════════════════════════════════════════════════════
def save_checkpoint(
    model: GlobalMultiHorizonGRU,
    path: Optional[str] = None,
) -> pathlib.Path:
    """Lưu trọng số mô hình ra file .pt"""
    if path is None:
        path = ROOT / "src" / "dl_gru" / "gru_best.pt"
    else:
        path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[Save] Checkpoint saved → {path}")
    return path


def load_checkpoint(
    model: GlobalMultiHorizonGRU,
    path: Optional[str] = None,
    device: torch.device = torch.device("cpu"),
) -> GlobalMultiHorizonGRU:
    """Nạp trọng số từ file .pt vào mô hình."""
    if path is None:
        path = ROOT / "src" / "dl_gru" / "gru_best.pt"
    else:
        path = pathlib.Path(path)
    model.load_state_dict(torch.load(path, map_location=device))
    model = model.to(device)
    print(f"[Load] Checkpoint loaded ← {path}")
    return model


# ══════════════════════════════════════════════════════════════
# 6.  PIPELINE CHÍNH (Huấn luyện → Dự báo → Lưu .npy)
# ══════════════════════════════════════════════════════════════
def run_full_pipeline(
    # Hyperparameters
    lookback: int = 30,
    horizon: int = 16,
    hidden_dim: int = 128,
    n_gru_layers: int = 2,
    emb_dim_store: int = 8,
    emb_dim_family: int = 8,
    emb_dim_cluster: int = 4,
    dropout: float = 0.2,
    batch_size: int = 1024,
    n_epochs: int = 60,
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    patience: int = 8,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Chạy toàn bộ pipeline:
      1. Tạo DataLoader (từ dataset.py)
      2. Khởi tạo mô hình GRU
      3. Huấn luyện với EarlyStopping
      4. Dự báo trên Validation & Test
      5. Lưu kết quả ra .npy

    Returns
    -------
    best_val_rmsle : float
    val_preds      : ndarray (N_val, 16) trên thang log1p
    test_preds     : ndarray (N_test, 16) trên thang log1p
    """
    # Import dataset builder (cùng package)
    from dataset import build_dataloaders

    print("=" * 70)
    print("  GLOBAL MULTI-HORIZON GRU – PIPELINE")
    print("=" * 70)

    # ---- Device ----
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[Device] GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[Device] CPU (cảnh báo: huấn luyện sẽ chậm)")

    # ---- 1. Data ----
    print("\n── 1. Xây dựng DataLoader ──")
    train_loader, val_loader, test_loader, cat_maps, dynamic_cols = \
        build_dataloaders(
            lookback=lookback,
            horizon=horizon,
            batch_size=batch_size,
        )

    n_dynamic = len(dynamic_cols)

    # ---- 2. Model ----
    print("\n── 2. Khởi tạo mô hình GRU ──")
    model = GlobalMultiHorizonGRU(
        n_dynamic=n_dynamic,
        hidden_dim=hidden_dim,
        n_gru_layers=n_gru_layers,
        horizon=horizon,
        n_stores=len(cat_maps["store_nbr"]),
        n_families=len(cat_maps["family"]),
        n_clusters=len(cat_maps["cluster"]),
        emb_dim_store=emb_dim_store,
        emb_dim_family=emb_dim_family,
        emb_dim_cluster=emb_dim_cluster,
        dropout=dropout,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Tổng tham số: {total_params:,}")
    print(f"  Input dynamic features: {n_dynamic}")
    print(f"  Hidden dim: {hidden_dim}, GRU layers: {n_gru_layers}")
    print(f"  Embedding dims: store={emb_dim_store}, "
          f"family={emb_dim_family}, cluster={emb_dim_cluster}")

    # ---- 3. Train ----
    print("\n── 3. Huấn luyện ──")
    history = train_model(
        model, train_loader, val_loader, device,
        n_epochs=n_epochs, lr=lr,
        weight_decay=weight_decay, patience=patience,
    )

    best_val_rmsle = min(history["val_loss"])

    # Vẽ và lưu biểu đồ Loss Curve phục vụ Báo cáo & Slide
    loss_curve_path = FIGURES / "gru_loss_curve.png"
    plot_loss_curve(history, loss_curve_path)

    # ---- 4. Predict & Hậu xử lý ----
    print("\n── 4. Sinh dự báo & Hậu xử lý ──")
    val_preds  = predict(model, val_loader, device)    # (N_val, 16)
    test_preds = predict(model, test_loader, device)   # (N_test, 16)

    # Áp dụng hậu xử lý (ép số âm về 0 & xử lý chuỗi toàn 0)
    zero_path  = PROC / "zero_series.csv"
    val_preds  = apply_postprocessing(val_preds, zero_path)
    test_preds = apply_postprocessing(test_preds, zero_path)

    print(f"  Val  predictions shape: {val_preds.shape}")
    print(f"  Test predictions shape: {test_preds.shape}")

    # ---- 5. Save ----
    print("\n── 5. Lưu kết quả ──")
    val_path  = PRED_VAL / "val_pred_gru.npy"
    test_path = PRED_TEST / "test_pred_gru.npy"
    np.save(val_path, val_preds)
    np.save(test_path, test_preds)
    print(f"  ✓ {val_path}")
    print(f"  ✓ {test_path}")

    # Lưu checkpoint
    save_checkpoint(model)

    print("\n" + "=" * 70)
    print(f"  HOÀN TẤT │ Best Val RMSLE = {best_val_rmsle:.5f}")
    print("=" * 70)

    return best_val_rmsle, val_preds, test_preds


# ══════════════════════════════════════════════════════════════
# 7.  ENTRYPOINT
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    best, vp, tp = run_full_pipeline()
