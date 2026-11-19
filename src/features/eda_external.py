"""
Owner: An Đức Minh Trí (24021645) - Task T03 (EDA)
EDA cho dữ liệu ngoại sinh: giá dầu, ngày lễ (đã khớp địa lý), ngày trong tuần, ngày trong tháng.
Chạy sau `python -m src.features.external`:

    python -m src.features.eda_external
    python -m src.features.eda_external --fig-dir <thư mục> --summary <file.md>   # không ghi đè reports/

Đầu ra mặc định: reports/figures/eda_*.png và reports/eda_external_summary.md

Phương pháp: doanh số tổng theo (store, ngày) trên thang log; "mức nền" = trung bình log của
cùng cửa hàng trong cùng tháng (khử quy mô + xu hướng). Cột "đã khử thứ" trừ thêm hiệu ứng
thứ trong tuần của từng cửa hàng, ước lượng trên ngày thường.
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .external import OUT_DIR, RAW_DIR, ROOT  # noqa: E402

FIG_DIR = ROOT / "reports" / "figures"
SUMMARY_PATH = ROOT / "reports" / "eda_external_summary.md"
DOW_NAMES = ["T2", "T3", "T4", "T5", "T6", "T7", "CN"]
SMALL_SAMPLE_DATES = 5


def pct(x):
    """Trung bình chênh lệch log -> phần trăm."""
    return float(np.expm1(x) * 100)


def load_store_day(raw_dir: Path, proc_dir: Path) -> pd.DataFrame:
    """Doanh số tổng theo (date, store_nbr) ghép với lịch lễ + giá dầu."""
    train = pd.read_csv(raw_dir / "train.csv", usecols=["date", "store_nbr", "sales"],
                        parse_dates=["date"])
    sd = train.groupby(["date", "store_nbr"], as_index=False)["sales"].sum()
    cal = pd.read_parquet(proc_dir / "holidays_store.parquet")
    oil = pd.read_parquet(proc_dir / "oil_clean.parquet")
    sd = sd.merge(cal, on=["date", "store_nbr"], how="left").merge(oil, on="date", how="left")

    # Bỏ ngày đóng cửa (1/1) và giai đoạn cửa hàng chưa mở (doanh số 0)
    sd = sd[(sd["sales"] > 0) & ~((sd["date"].dt.month == 1) & (sd["date"].dt.day == 1))].copy()
    sd["log_sales"] = np.log1p(sd["sales"])
    ym = sd["date"].dt.to_period("M")
    sd["log_ratio"] = sd["log_sales"] - sd.groupby(["store_nbr", ym])["log_sales"].transform("mean")

    d = sd["date"].dt
    sd["dayofweek"] = d.day_of_week
    sd["day_of_month"] = d.day
    sd["is_payday"] = ((d.day == 15) | d.is_month_end).astype(int)
    # Số ngày kể từ lần nhận lương gần nhất (ngày 15 hoặc ngày cuối tháng trước)
    sd["days_since_payday"] = np.where(d.is_month_end, 0,
                                       np.where(d.day >= 15, d.day - 15, d.day))

    # Ngày thường = không có bất kỳ cờ lịch đặc biệt nào
    sd["is_normal"] = ((sd["is_holiday"] == 0) & (sd["is_additional"] == 0)
                       & (sd["is_event"] == 0) & (sd["is_workday"] == 0)).astype(int)
    dow_eff = (sd[sd["is_normal"] == 1].groupby(["store_nbr", "dayofweek"])["log_ratio"]
               .mean().rename("dow_eff"))
    sd = sd.join(dow_eff, on=["store_nbr", "dayofweek"])
    sd["log_ratio_adj"] = sd["log_ratio"] - sd["dow_eff"]
    return sd


# --------------------------------------------------------------------------- oil
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


def detrend(y: pd.Series) -> pd.Series:
    t = np.arange(len(y))
    return y - np.polyval(np.polyfit(t, y.to_numpy(), 1), t)


def oil_correlations(sd: pd.DataFrame) -> dict:
    """Tương quan doanh số - giá dầu: mức thô, sau khử xu hướng tuyến tính, và theo % thay đổi tháng."""
    daily = sd.groupby("date").agg(sales=("sales", "sum"), oil=("oil", "first"))
    roll = daily["sales"].rolling(28).mean()
    ok = roll.notna()
    monthly = daily.resample("MS").mean().pct_change().dropna()
    return {
        "daily": daily,
        "raw": roll.corr(daily["oil"]),
        "detrended": detrend(roll[ok]).corr(detrend(daily.loc[ok, "oil"])),
        "monthly_pct": monthly["sales"].corr(monthly["oil"]),
    }


def fig_sales_vs_oil(corr: dict, path: Path):
    daily = corr["daily"]
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(daily.index, daily["sales"].rolling(28, min_periods=1).mean() / 1e3,
            color="tab:green", label="Tổng doanh số (TB trượt 28 ngày, nghìn)")
    ax.set_ylabel("Doanh số (nghìn)")
    ax2 = ax.twinx()
    ax2.plot(daily.index, daily["oil"], color="tab:blue", lw=0.8, label="Giá dầu")
    ax2.set_ylabel("USD/thùng")
    ax.set_title(f"Doanh số tổng và giá dầu - tương quan thô {corr['raw']:.2f}, "
                 f"sau khử xu hướng {corr['detrended']:.2f}, "
                 f"theo % thay đổi tháng {corr['monthly_pct']:.2f}")
    fig.legend(loc="center left", bbox_to_anchor=(0.07, 0.55))
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------- holidays
def holiday_effect(sd: pd.DataFrame) -> pd.DataFrame:
    groups = {
        "Ngày thường": sd["is_normal"] == 1,
        "Lễ National (nghỉ)": sd["hol_national"] == 1,
        "Lễ Regional (nghỉ)": sd["hol_regional"] == 1,
        "Lễ Local (nghỉ)": sd["hol_local"] == 1,
        "Bridge": sd["is_bridge"] == 1,
        "Additional (quanh Giáng sinh)": sd["is_additional"] == 1,
        "Work Day": sd["is_workday"] == 1,
        "Event: động đất": sd["is_earthquake"] == 1,
        "Event: mua sắm": sd["is_shopping_event"] == 1,
        "Event: World Cup": sd["is_sport_event"] == 1,
    }
    rows = []
    for name, m in groups.items():
        n_dates = sd.loc[m, "date"].nunique()
        if n_dates < SMALL_SAMPLE_DATES:
            name += " (mẫu nhỏ)"
        rows.append((name, n_dates, int(m.sum()),
                     pct(sd.loc[m, "log_ratio"].mean()), pct(sd.loc[m, "log_ratio_adj"].mean())))
    return pd.DataFrame(rows, columns=["nhóm", "số ngày", "số (store, ngày)",
                                       "% so với TB tháng", "% đã khử thứ"])


def fig_holiday(eff: pd.DataFrame, path: Path):
    fig, ax = plt.subplots(figsize=(9, 4.6))
    vals = eff["% đã khử thứ"]
    ax.barh(eff["nhóm"], vals, color=np.where(vals >= 0, "tab:green", "tab:red"))
    ax.axvline(0, color="k", lw=0.8)
    ax.invert_yaxis()
    ax.set_xlabel("% chênh lệch doanh số (đã khử quy mô cửa hàng, tháng, thứ trong tuần)")
    ax.set_title("Tác động của ngày lễ / sự kiện (đã khớp địa lý, bỏ ngày lễ bị dời)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ------------------------------------------------------------- weekday / payday
def calendar_effects(sd: pd.DataFrame) -> dict:
    normal = sd[sd["is_normal"] == 1]
    return {
        "dow": normal.groupby("dayofweek")["log_ratio"].mean().map(pct),
        "dom": normal.groupby("day_of_month")["log_ratio_adj"].mean().map(pct),
        "since_payday": normal.groupby("days_since_payday")["log_ratio_adj"].mean().map(pct),
        "payday": normal.groupby("is_payday")["log_ratio_adj"].mean().map(pct),
    }


def fig_dow_dom(eff: dict, path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1, 2.4]})
    axes[0].bar(DOW_NAMES, eff["dow"].values, color="tab:blue")
    axes[0].axhline(0, color="k", lw=0.8)
    axes[0].set_title("Theo thứ trong tuần (%)")
    dom = eff["dom"]
    colors = ["tab:orange" if d in (15, 30, 31) else "tab:gray" for d in dom.index]
    axes[1].bar(dom.index, dom.values, color=colors)
    axes[1].axhline(0, color="k", lw=0.8)
    axes[1].set_xticks(range(1, 32, 2))
    axes[1].set_title("Theo ngày trong tháng, đã khử thứ (%) - cam: ngày lương 15 & cuối tháng")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------- run
def md_table(df: pd.DataFrame) -> list:
    return ["| " + " | ".join(df.columns) + " |",
            "|" + " --- |" * len(df.columns),
            *("| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False))]


def run(raw_dir=RAW_DIR, proc_dir=OUT_DIR, fig_dir=FIG_DIR, summary_path=SUMMARY_PATH):
    raw_dir, proc_dir, fig_dir = Path(raw_dir), Path(proc_dir), Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    oil = pd.read_parquet(proc_dir / "oil_clean.parquet")
    sd = load_store_day(raw_dir, proc_dir)

    fig_oil(oil, fig_dir / "eda_oil_interpolation.png")
    corr = oil_correlations(sd)
    fig_sales_vs_oil(corr, fig_dir / "eda_sales_vs_oil.png")
    eff = holiday_effect(sd)
    fig_holiday(eff, fig_dir / "eda_holiday_effect.png")
    cal = calendar_effects(sd)
    fig_dow_dom(cal, fig_dir / "eda_dayofweek_payday.png")

    eff_fmt = eff.copy()
    for col in eff.columns[3:]:
        eff_fmt[col] = eff[col].map("{:+.1f}".format)
    dom, since = cal["dom"], cal["since_payday"]
    lines = [
        "# T03 - EDA dữ liệu ngoại sinh",
        "",
        "Sinh tự động bởi `python -m src.features.eda_external`. Mức nền = trung bình log doanh số "
        "của cùng cửa hàng trong cùng tháng; \"% đã khử thứ\" trừ thêm hiệu ứng thứ trong tuần của "
        "từng cửa hàng (ước lượng trên ngày thường).",
        "",
        "## Giá dầu",
        "",
        f"- {int(oil['dcoilwtico'].isna().sum())} / {len(oil)} ngày thiếu giá được nội suy tuyến tính "
        "(chủ yếu cuối tuần).",
        f"- Tương quan tổng doanh số (TB 28 ngày) với giá dầu: thô **{corr['raw']:.2f}**, "
        f"sau khử xu hướng tuyến tính **{corr['detrended']:.2f}**, "
        f"theo % thay đổi hằng tháng **{corr['monthly_pct']:.2f}**.",
        "- Tương quan thô chủ yếu do hai chuỗi cùng có xu hướng theo thời gian (doanh số tăng, "
        "dầu giảm từ 2014-2015); sau khi khử xu hướng chỉ còn quan hệ yếu và theo biến động "
        "ngắn hạn thì gần như bằng 0. Không nên kỳ vọng nhiều vào `oil_ma*`.",
        "",
        "## Tác động ngày lễ / sự kiện",
        "",
        *md_table(eff_fmt),
        "",
        f"\"(mẫu nhỏ)\": ít hơn {SMALL_SAMPLE_DATES} ngày khác nhau, con số kém tin cậy. "
        "Work Day thường rơi vào thứ 7 nên chỉ cột \"đã khử thứ\" là có ý nghĩa.",
        "",
        "## Thứ trong tuần (ngày thường, %)",
        "",
        " | ".join(f"{d}: {v:+.1f}" for d, v in zip(DOW_NAMES, cal["dow"])),
        "",
        "## Ngày trong tháng / ngày lương (ngày thường, đã khử thứ, %)",
        "",
        "- Theo ngày: " + ", ".join(f"ngày {d}: {dom[d]:+.1f}" for d in (1, 2, 3, 4, 5, 15, 16, 30, 31)),
        "- Theo số ngày kể từ ngày lương: "
        + ", ".join(f"{k}: {since[k]:+.1f}" for k in range(0, 7)),
        f"- Cờ `is_payday` (ngày 15 + cuối tháng): {cal['payday'].get(1, np.nan):+.1f}% "
        f"(ngày khác: {cal['payday'].get(0, np.nan):+.1f}%).",
        "- Đỉnh doanh số rơi vào **ngày 1-3 đầu tháng** (sau ngày lương cuối tháng) và giảm dần; "
        "bản thân ngày 15 / cuối tháng gần như không tăng. Gợi ý T05: dùng `day_of_month`, "
        "`days_since_payday` thay vì chỉ `is_payday`.",
        "",
        "Biểu đồ: `reports/figures/eda_oil_interpolation.png`, `eda_sales_vs_oil.png`, "
        "`eda_holiday_effect.png`, `eda_dayofweek_payday.png`.",
    ]
    Path(summary_path).parent.mkdir(parents=True, exist_ok=True)
    Path(summary_path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    # Console Windows (cp1252) không in được tiếng Việt
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir", default=RAW_DIR)
    p.add_argument("--proc-dir", default=OUT_DIR)
    p.add_argument("--fig-dir", default=FIG_DIR)
    p.add_argument("--summary", default=SUMMARY_PATH)
    a = p.parse_args()
    run(a.raw_dir, a.proc_dir, a.fig_dir, a.summary)
