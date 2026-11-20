"""Feature groups for the XGBoost plan; sales and transactions end at t-16."""

import numpy as np
import pandas as pd

FEATURE_GROUPS = {
    "same_weekday": ["dow_mean_4", "dow_mean_8"],
    "sparse": ["zero_fraction_14", "zero_fraction_28", "days_since_sale_16"],
    "long_history": ["rolling_mean_14", "rolling_mean_56", "rolling_mean_112",
                     "ewm_sales_7", "ewm_sales_28"],
    "yearly": ["sales_lag_364_mean_7"],
    "cross_section": ["store_mean_log_lag16", "family_mean_log_lag16",
                      "city_family_mean_log_lag16"],
    "promotion": ["promo_lag_1", "promo_lag_7", "promo_lead_mean_3",
                  "promo_lead_mean_7", "promo_ratio_ma28"],
    "transactions": ["transactions_mean_7_lag16", "transactions_mean_28_lag16"],
}
GUIDED_FEATURES = [c for group in FEATURE_GROUPS.values() for c in group]


def _cross_mean(df, keys, name):
    grouped = df.groupby(["date", *keys], observed=True, sort=False)["sales_log1p"].mean()
    lookup = df[["date", *keys]].copy()
    lookup["date"] = lookup["date"] - pd.Timedelta(days=16)
    index = pd.MultiIndex.from_frame(lookup)
    df[name] = grouped.reindex(index).to_numpy(dtype="float32")


def _transactions(df, transactions):
    raw = transactions[["date", "store_nbr", "transactions"]].copy()
    raw["date"] = pd.to_datetime(raw["date"])
    if raw.duplicated(["date", "store_nbr"]).any():
        raise ValueError("Duplicate transaction dates for a store")
    grid = pd.MultiIndex.from_product(
        [np.sort(df["store_nbr"].astype("int16").unique()),
         pd.date_range(df["date"].min(), df["date"].max())],
        names=["store_nbr", "date"],
    )
    daily = raw.set_index(["store_nbr", "date"])["transactions"].reindex(grid)
    delayed = daily.groupby(level=0, sort=False).shift(16)
    lookup = pd.MultiIndex.from_arrays([df["store_nbr"].astype("int16"), df["date"]])
    for w, minimum in ((7, 4), (28, 14)):
        mean = delayed.groupby(level=0, sort=False).transform(
            lambda x: x.rolling(w, min_periods=minimum).mean())
        df[f"transactions_mean_{w}_lag16"] = mean.reindex(lookup).to_numpy(dtype="float32")


def add_guided_features(df, transactions):
    """Add every group in place to a continuous, series/date sorted frame."""
    sales = df.groupby("series_id", sort=False, observed=True)["sales_log1p"]
    lags = {k: sales.shift(k).astype("float32") for k in (21, 28, 35, 42, 49, 56, 63, 70)}
    df["dow_mean_4"] = pd.concat([lags[k] for k in (21, 28, 35, 42)], axis=1).mean(axis=1).astype("float32")
    df["dow_mean_8"] = pd.concat(list(lags.values()), axis=1).mean(axis=1).astype("float32")
    del lags

    base = sales.shift(16)
    grouped = base.groupby(df["series_id"], sort=False, observed=True)
    zeros = base.eq(0).where(base.notna()).astype("float32")
    zgroup = zeros.groupby(df["series_id"], sort=False, observed=True)
    for window, minimum in ((14, 7), (28, 14)):
        df[f"zero_fraction_{window}"] = zgroup.transform(
            lambda x: x.rolling(window, min_periods=minimum).mean()).astype("float32")
    observed_date = (df["date"] - pd.Timedelta(days=16)).where(base.gt(0))
    last = observed_date.groupby(df["series_id"], sort=False, observed=True).ffill()
    df["days_since_sale_16"] = ((df["date"] - pd.Timedelta(days=16)) - last).dt.days.astype("float32")
    for window, minimum in ((14, 7), (56, 28), (112, 56)):
        df[f"rolling_mean_{window}"] = grouped.transform(
            lambda x: x.rolling(window, min_periods=minimum).mean()).astype("float32")
    for span in (7, 28):
        df[f"ewm_sales_{span}"] = grouped.transform(
            lambda x: x.ewm(span=span, adjust=False, ignore_na=True).mean()).astype("float32")
    year_lags = [sales.shift(k) for k in range(361, 368)]
    df["sales_lag_364_mean_7"] = pd.concat(year_lags, axis=1).mean(axis=1).astype("float32")

    _cross_mean(df, ["store_nbr"], "store_mean_log_lag16")
    _cross_mean(df, ["family"], "family_mean_log_lag16")
    _cross_mean(df, ["city", "family"], "city_family_mean_log_lag16")

    promo = df.groupby("series_id", sort=False, observed=True)["onpromotion"]
    for lag in (1, 7):
        df[f"promo_lag_{lag}"] = promo.shift(lag).astype("float32")
    for horizon in (3, 7):
        leads = pd.concat([promo.shift(-k) for k in range(1, horizon + 1)], axis=1)
        df[f"promo_lead_mean_{horizon}"] = leads.mean(axis=1).fillna(df["onpromotion"]).astype("float32")
    ma28 = promo.transform(lambda x: x.rolling(28, min_periods=1).mean())
    df["promo_ratio_ma28"] = (df["onpromotion"] / (1 + ma28)).astype("float32")
    _transactions(df, transactions)
    return df


def restrict_promo_leads(frame):
    """Recompute leads within a fold; promotions after the horizon are unknown."""
    frame = frame.copy()
    promo = frame.groupby("series_id", sort=False, observed=True)["onpromotion"]
    if "onpromotion_lead1" in frame.columns:
        frame["onpromotion_lead1"] = promo.shift(-1).fillna(frame["onpromotion"]).astype("float32")
    for horizon in (3, 7):
        name = f"promo_lead_mean_{horizon}"
        if name in frame.columns:
            leads = pd.concat([promo.shift(-k) for k in range(1, horizon + 1)], axis=1)
            frame[name] = leads.mean(axis=1).fillna(frame["onpromotion"]).astype("float32")
    return frame
