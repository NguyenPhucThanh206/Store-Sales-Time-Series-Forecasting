"""
Owner: An Đức Minh Trí (24021645) - Task T03 (EDA)
EDA cho dữ liệu ngoại sinh: giá dầu, ngày lễ (đã khớp địa lý), ngày trong tuần, ngày lương.
Chạy sau `python -m src.features.external`:

    python -m src.features.eda_external

Đầu ra: reports/figures/eda_*.png và reports/eda_external_summary.md
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .external import OUT_DIR, RAW_DIR, ROOT  # noqa: E402

FIG_DIR = ROOT / "reports" / "figures"
SUMMARY_PATH = ROOT / "reports" / "eda_external_summary.md"


def load_store_day(raw_dir: Path, proc_dir: Path) -> pd.DataFrame:
    """Doanh số tổng theo (date, store_nbr) ghép với lịch lễ + giá dầu."""
    train = pd.read_csv(raw_dir / "train.csv", usecols=["date", "store_nbr", "sales"],
                        parse_dates=["date"])
    sd = train.groupby(["date", "store_nbr"], as_index=False)["sales"].sum()
    cal = pd.read_parquet(proc_dir / "holidays_store.parquet")
    oil = pd.read_parquet(proc_dir / "oil_clean.parquet")
    sd = sd.merge(cal, on=["date", "store_nbr"], how="left").merge(oil, on="date", how="left")

    # Bỏ ngày đóng cửa (1/1) và giai đoạn cửa hàng chưa mở (doanh số 0)
    sd = sd[(sd["sales"] > 0) & ~((sd["date"].dt.month == 1) & (sd["date"].dt.day == 1))]
    sd["log_sales"] = np.log1p(sd["sales"])
    # Mức nền: trung bình log doanh số của cửa hàng trong cùng tháng -> loại xu hướng & quy mô
    ym = sd["date"].dt.to_period("M")
    sd["log_ratio"] = sd["log_sales"] - sd.groupby(["store_nbr", ym])["log_sales"].transform("mean")
    sd["dayofweek"] = sd["date"].dt.day_of_week
    sd["is_payday"] = ((sd["date"].dt.day == 15) | sd["date"].dt.is_month_end).astype(int)
    # Khử thêm hiệu ứng thứ trong tuần (ước lượng trên ngày thường) để so sánh ngày lễ công bằng
    normal = (sd["is_holiday"] == 0) & (sd["is_event"] == 0) & (sd["is_workday"] == 0)
    dow_eff = sd[normal].groupby(["store_nbr", "dayofweek"])["log_ratio"].mean().rename("dow_eff")
    sd = sd.join(dow_eff, on=["store_nbr", "dayofweek"])
    sd["log_ratio_adj"] = sd["log_ratio"] - sd["dow_eff"]
    return sd


def fig_oil(oil: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(oil["date"], oil["oil"], lw=0.8, color="tab:blue", label="oil (nội suy)")
    ax.plot(oil["date"], oil["oil_ma14"], lw=1.2, color="tab:orange", label="oil_ma14")
    miss = oil["dcoilwtico"].isna()
    ax.scatter(oil.loc[miss, "date"], oil.loc[miss, "oil"], s=2, color="tab:red",
               label=f"điểm nội suy ({int(miss.sum())})")
    ax.set_title("Giá dầu WTI sau nội suy tuyến tính")
    ax.set_ylabel("USD/thùng")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_sales_vs_oil(sd: pd.DataFrame, path: Path):
    daily = sd.groupby("date").agg(sales=("sales", "sum"), oil=("oil", "first"))
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily.index, daily["sales"].rolling(28, min_periods=1).mean() / 1e3,
            color="tab:green", label="Tổng doanh số (TB trượt 28 ngày, nghìn)")
    ax.set_ylabel("Doanh số (nghìn)")
    ax2 = ax.twinx()
    ax2.plot(daily.index, daily["oil"], color="tab:blue", lw=0.8, label="Giá dầu")
    ax2.set_ylabel("USD/thùng")
    corr = daily["sales"].rolling(28).mean().corr(daily["oil"])
    ax.set_title(f"Doanh số tổng và giá dầu (tương quan = {corr:.2f})")
    fig.legend(loc="upper left", bbox_to_anchor=(0.07, 0.88))
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return corr


def holiday_effect(sd: pd.DataFrame) -> pd.DataFrame:
    groups = {
        "Ngày thường": (sd["is_holiday"] == 0) & (sd["is_event"] == 0) & (sd["is_workday"] == 0),
        "Lễ National": sd["hol_national"] == 1,
        "Lễ Regional": sd["hol_regional"] == 1,
        "Lễ Local": sd["hol_local"] == 1,
        "Bridge": sd["is_bridge"] == 1,
        "Additional": sd["is_additional"] == 1,
        "Event": sd["is_event"] == 1,
        "Work Day": sd["is_workday"] == 1,
    }
    rows = [(k, int(m.sum()), np.expm1(sd.loc[m, "log_ratio"].mean()) * 100,
             np.expm1(sd.loc[m, "log_ratio_adj"].mean()) * 100)
            for k, m in groups.items()]
    return pd.DataFrame(rows, columns=["nhóm", "số (store, ngày)", "chênh lệch % so với TB tháng",
                                       "chênh lệch % (đã khử thứ trong tuần)"])


def fig_holiday(eff: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(9, 4))
    vals = eff["chênh lệch % (đã khử thứ trong tuần)"]
    ax.barh(eff["nhóm"], vals, color=np.where(vals >= 0, "tab:green", "tab:red"))
    ax.axvline(0, color="k", lw=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("% chênh lệch doanh số (đã khử quy mô cửa hàng, tháng, thứ trong tuần)")
    ax.set_title("Tác động của ngày lễ (đã khớp địa lý, bỏ ngày lễ bị dời)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_dow_payday(sd: pd.DataFrame, path: Path):
    normal = sd[(sd["is_holiday"] == 0) & (sd["is_event"] == 0)]
    dow = np.expm1(normal.groupby("dayofweek")["log_ratio"].mean()) * 100
    pay = np.expm1(normal.groupby("is_payday")["log_ratio"].mean()) * 100
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), gridspec_kw={"width_ratios": [3, 1]})
    axes[0].bar(["T2", "T3", "T4", "T5", "T6", "T7", "CN"], dow.values, color="tab:blue")
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_title("Chênh lệch doanh số theo ngày trong tuần (%)")
    axes[1].bar(["Thường", "Ngày lương"], pay.values, color=["tab:gray", "tab:orange"])
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_title("Ngày 15 & cuối tháng (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return dow, pay


def run(raw_dir=RAW_DIR, proc_dir=OUT_DIR, fig_dir=FIG_DIR, summary_path=SUMMARY_PATH):
    raw_dir, proc_dir, fig_dir = Path(raw_dir), Path(proc_dir), Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    oil = pd.read_parquet(proc_dir / "oil_clean.parquet")
    sd = load_store_day(raw_dir, proc_dir)

    fig_oil(oil, fig_dir / "eda_oil_interpolation.png")
    corr = fig_sales_vs_oil(sd, fig_dir / "eda_sales_vs_oil.png")
    eff = holiday_effect(sd)
    fig_holiday(eff, fig_dir / "eda_holiday_effect.png")
    dow, pay = fig_dow_payday(sd, fig_dir / "eda_dayofweek_payday.png")

    eff_fmt = eff.copy()
    for col in eff.columns[2:]:
        eff_fmt[col] = eff[col].map("{:+.1f}".format)
    lines = [
        "# T03 - EDA dữ liệu ngoại sinh",
        "",
        "Sinh tự động bởi `python -m src.features.eda_external`.",
        "",
        f"- Giá dầu: {int(oil['dcoilwtico'].isna().sum())} / {len(oil)} ngày thiếu giá "
        "được nội suy tuyến tính (chủ yếu cuối tuần).",
        f"- Tương quan giữa tổng doanh số (TB 28 ngày) và giá dầu: **{corr:.2f}**.",
        "",
        "## Tác động ngày lễ (so với trung bình tháng của cùng cửa hàng)",
        "",
        "| " + " | ".join(eff_fmt.columns) + " |",
        "|" + " --- |" * len(eff_fmt.columns),
        *("| " + " | ".join(map(str, r)) + " |" for r in eff_fmt.itertuples(index=False)),
        "",
        "## Ngày trong tuần (ngày thường, %)",
        "",
        " | ".join(f"{d}: {v:+.1f}" for d, v in zip(["T2", "T3", "T4", "T5", "T6", "T7", "CN"], dow)),
        "",
        f"## Ngày lương: {pay.get(1, np.nan):+.1f}% (ngày khác: {pay.get(0, np.nan):+.1f}%)",
        "",
        "Biểu đồ: `reports/figures/eda_oil_interpolation.png`, `eda_sales_vs_oil.png`, "
        "`eda_holiday_effect.png`, `eda_dayofweek_payday.png`.",
    ]
    Path(summary_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default=RAW_DIR)
    p.add_argument("--proc-dir", default=OUT_DIR)
    a = p.parse_args()
    run(a.raw_dir, a.proc_dir)
