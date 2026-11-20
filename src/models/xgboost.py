"""T07: train XGBoost on the shared feature set and export log1p predictions.

Run from the repository root after T02, T03 and T05::

    python -m src.models.xgboost

The validation and test arrays follow the row order of ``train_fe[is_val == 1]``
and ``test_fe`` respectively (series_id, date). Both arrays stay on the
log1p(sales) scale for blending; submission conversion happens elsewhere.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.features.build_features import (
    CAT_FEATURES,
    FEATURES,
    PROC_DIR,
    ROOT,
    TARGET,
    apply_zero_series,
    load_fe,
)


def _normalize_numeric_categories(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Give integer categories a numeric Index (needed by XGBoost/Pandas 3).

    T05's load_fe() may restore store_nbr and cluster with an object Index of
    Python integers. XGBoost rejects that Index even with enable_categorical.
    """
    for col in ("store_nbr", "cluster"):
        categories = pd.Index(train[col].cat.categories.to_numpy(dtype="int64"))
        for frame in (train, test):
            frame[col] = pd.Categorical(frame[col], categories=categories)


def _check_inputs(train: pd.DataFrame, test: pd.DataFrame, horizon: int = 16) -> None:
    """Reject input that would produce misaligned or incorrectly typed predictions."""
    train_cols = {*FEATURES, TARGET, "is_val", "series_id", "date"}
    test_cols = {*FEATURES, "series_id", "date", "id", "is_zero_series"}
    for name, frame, required in (("train_fe", train, train_cols),
                                  ("test_fe", test, test_cols)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{name} missing columns: {missing}")
        if frame.duplicated(["series_id", "date"]).any():
            raise ValueError(f"{name} has duplicate (series_id, date) rows")
        ordered = frame.sort_values(["series_id", "date"]).index.equals(frame.index)
        if not ordered:
            raise ValueError(f"{name} must be sorted by (series_id, date)")

    if test["id"].duplicated().any():
        raise ValueError("test_fe contains duplicate Kaggle ids")
    if not train["is_val"].isin([0, 1]).all():
        raise ValueError("is_val must contain only 0 and 1")
    if train.loc[train["is_val"] == 1, "date"].nunique() != horizon:
        raise ValueError(f"validation must contain {horizon} dates")
    if train.loc[train["is_val"] == 1].groupby("series_id").size().ne(horizon).any():
        raise ValueError(f"each validation series must contain {horizon} rows")
    if test.groupby("series_id").size().ne(horizon).any():
        raise ValueError(f"each test series must contain {horizon} rows")
    for col in CAT_FEATURES:
        if not isinstance(train[col].dtype, pd.CategoricalDtype):
            raise TypeError(f"{col} must have category dtype for XGBoost")
        if not train[col].cat.categories.equals(test[col].cat.categories):
            raise ValueError(f"{col} categories differ between train and test")


def run(
    proc_dir: Path = PROC_DIR,
    predictions_dir: Path = ROOT / "predictions",
    n_estimators: int = 1200,
    early_stopping_rounds: int = 100,
    n_jobs: int = -1,
) -> dict:
    """Fit on pre-validation rows and write ordered validation/test predictions."""
    proc_dir = Path(proc_dir)
    predictions_dir = Path(predictions_dir)
    for filename in ("train_fe.parquet", "test_fe.parquet"):
        if not (proc_dir / filename).exists():
            raise FileNotFoundError(
                f"Missing {proc_dir / filename}; run src.features.build_features first"
            )
    if n_estimators < 1 or early_stopping_rounds < 1:
        raise ValueError("n_estimators and early_stopping_rounds must be positive")

    train, test = load_fe(proc_dir)
    _normalize_numeric_categories(train, test)
    _check_inputs(train, test)

    train_mask = (train["is_val"] == 0) & train[TARGET].notna()
    val_mask = train["is_val"] == 1
    val_scored = train.loc[val_mask, TARGET].notna().to_numpy()
    if not train_mask.any() or not val_scored.any():
        raise ValueError("No observed train or validation targets")

    x_train = train.loc[train_mask, FEATURES]
    y_train = train.loc[train_mask, TARGET].astype("float32")
    x_val = train.loc[val_mask, FEATURES]
    y_val = train.loc[val_mask, TARGET].astype("float32")
    x_test = test.loc[:, FEATURES]

    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        enable_categorical=True,
        grow_policy="depthwise",
        max_depth=8,
        learning_rate=0.05,
        min_child_weight=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=10.0,
        max_cat_to_onehot=8,
        n_estimators=n_estimators,
        early_stopping_rounds=early_stopping_rounds,
        eval_metric="rmse",  # RMSE(log1p(sales)) is RMSLE before clipping.
        random_state=42,
        n_jobs=n_jobs,
    )
    model.fit(
        x_train, y_train,
        eval_set=[(x_val.loc[val_scored], y_val.loc[val_scored])],
        verbose=100,
    )

    # XGBoost uses best_iteration automatically after early stopping.
    val_pred = np.clip(model.predict(x_val).astype("float32"), 0.0, None)
    test_pred = np.clip(model.predict(x_test).astype("float32"), 0.0, None)
    test_pred = apply_zero_series(test_pred, test).astype("float32")
    if len(val_pred) != int(val_mask.sum()) or len(test_pred) != len(test):
        raise AssertionError("Prediction row count does not match feature tables")
    if not np.isfinite(val_pred).all() or not np.isfinite(test_pred).all():
        raise ValueError("XGBoost produced non-finite predictions")

    # y_val is already log1p(sales); clipping keeps exported values in-domain.
    val_rmsle = float(np.sqrt(np.mean(
        np.square(val_pred[val_scored] - y_val.to_numpy()[val_scored])
    )))

    val_path = predictions_dir / "val" / "val_pred_xgboost.npy"
    test_path = predictions_dir / "test" / "test_pred_xgboost.npy"
    model_path = predictions_dir / "models" / "xgboost.ubj"
    for path in (val_path, test_path, model_path):
        path.parent.mkdir(parents=True, exist_ok=True)
    np.save(val_path, val_pred)
    np.save(test_path, test_pred)
    # Save exactly the trees used by predict(), including for later SHAP analysis.
    model.get_booster()[:model.best_iteration + 1].save_model(str(model_path))

    print(f"Best iteration: {model.best_iteration}")
    print(f"Validation RMSLE: {val_rmsle:.6f} ({val_scored.sum():,} scored rows)")
    print(f"Saved: {val_path} ({val_pred.shape}), {test_path} ({test_pred.shape})")
    print(f"Model: {model_path}")
    return {"val_rmsle": val_rmsle, "best_iteration": model.best_iteration,
            "val_path": val_path, "test_path": test_path, "model_path": model_path}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proc-dir", type=Path, default=PROC_DIR)
    parser.add_argument("--predictions-dir", type=Path, default=ROOT / "predictions")
    parser.add_argument("--n-estimators", type=int, default=1200)
    parser.add_argument("--early-stopping-rounds", type=int, default=100)
    parser.add_argument("--n-jobs", type=int, default=-1)
    args = parser.parse_args()
    run(args.proc_dir, args.predictions_dir, args.n_estimators,
        args.early_stopping_rounds, args.n_jobs)
