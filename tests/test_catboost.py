"""T08: kiểm tra CatBoost trên bảng đặc trưng giả lập nhỏ (thứ tự, kích thước, ép chuỗi toàn 0)."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.features.build_features import CAT_FEATURES, FEATURES, TARGET
from src.models.catboost import check_inputs, kaggle_like_rmsle, run


def _fake_tables(n_series=6, n_train_days=80, seed=0):
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2017-01-01")
    cat_values = {"store_nbr": [1, 2, 3], "family": ["A", "B"], "city": ["Quito", "Cuenca"],
                  "state": ["Pichincha", "Azuay"], "store_type": ["A", "D"], "cluster": [1, 5]}

    def frame(days, offset):
        rows = []
        for sid in range(n_series):
            for d in range(days):
                rows.append({"series_id": sid, "date": start + pd.Timedelta(days=offset + d)})
        df = pd.DataFrame(rows)
        for c in FEATURES:
            if c in CAT_FEATURES:
                vals = cat_values[c]
                df[c] = [vals[sid % len(vals)] for sid in df["series_id"]]
            else:
                df[c] = rng.normal(size=len(df)).astype("float32")
        return df

    train = frame(n_train_days, 0)
    train[TARGET] = np.abs(train["sales_lag_16"]) + train["series_id"] * 0.5
    train["sales"] = np.expm1(train[TARGET])
    train["is_val"] = (train["date"] >= start + pd.Timedelta(days=n_train_days - 16)).astype("int8")
    train["is_closed"] = np.int8(0)
    test = frame(16, n_train_days)
    test["id"] = np.arange(len(test))[::-1]          # id không tăng dần, như test_fe thật
    test["is_zero_series"] = (test["series_id"] == 0).astype("int8")
    return train, test


class CatBoostChecks(unittest.TestCase):
    def test_run_writes_ordered_predictions(self):
        train, test = _fake_tables()
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            train.to_parquet(tmp / "train_fe.parquet", index=False)
            test.to_parquet(tmp / "test_fe.parquet", index=False)
            r = run(proc_dir=tmp, pred_dir=tmp / "pred", report_dir=tmp / "rep",
                    iterations=30, early_stopping=10, verbose=0)
            val = np.load(r["val"])
            tst = np.load(r["test"])
            self.assertEqual(val.shape, (int(train["is_val"].sum()),))
            self.assertEqual(tst.shape, (len(test),))
            self.assertEqual(val.dtype, np.float32)
            self.assertTrue((val >= 0).all() and (tst >= 0).all())
            np.testing.assert_array_equal(tst[test["is_zero_series"].to_numpy() == 1], 0)
            self.assertTrue((tmp / "pred" / "models" / "catboost.cbm").exists())
            self.assertTrue((tmp / "rep" / "catboost_results.md").exists())

    def test_rejects_unsorted_test(self):
        train, test = _fake_tables()
        for d in (train, test):
            for c in CAT_FEATURES:
                d[c] = d[c].astype("category")
        test = test.iloc[::-1].reset_index(drop=True)
        with self.assertRaises(ValueError):
            check_inputs(train, test, list(FEATURES))

    def test_kaggle_like_rmsle_adds_zero_rows(self):
        y, p = np.zeros(100), np.ones(100)
        self.assertAlmostEqual(kaggle_like_rmsle(y, p, n_zero_rows=300), 0.5)


if __name__ == "__main__":
    unittest.main()
