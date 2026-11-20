"""Small integration checks for the GRU windows and XGBoost handoff."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.dl_gru import dataset as gru
from src.features.build_features import FEATURES, TARGET
from src.models.xgboost import run as run_xgboost


class PipelineChecks(unittest.TestCase):
    def test_validation_keeps_short_series(self):
        start = pd.Timestamp("2017-01-01")
        frame = pd.DataFrame({
            "date": [start + pd.Timedelta(days=day) for day in range(36)],
            "store_nbr": 1,
            "family": "A",
            "log_sales": np.arange(36, dtype="float32"),
        })
        stores = pd.DataFrame({"store_nbr": [1], "cluster": [1]})
        maps = gru._build_category_maps(frame, stores)
        val_dates = [(start + pd.Timedelta(days=day)).strftime("%Y-%m-%d")
                     for day in range(20, 36)]
        samples = gru._prepare_series(frame, stores, maps, mode="val",
                                      val_dates=val_dates)
        self.assertEqual(len(samples), 1)
        np.testing.assert_array_equal(samples[0]["x_dynamic"][:10], 0)
        np.testing.assert_array_equal(samples[0]["x_dynamic"][10:, 0],
                                      np.arange(20))
        np.testing.assert_array_equal(samples[0]["y_target"], np.arange(20, 36))

    def test_windows_and_xgboost_outputs(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            raw, proc = root / "raw", root / "processed"
            raw.mkdir()
            proc.mkdir()
            pd.DataFrame({"store_nbr": [1, 2], "cluster": [1, 2]}).to_csv(
                raw / "stores.csv", index=False
            )

            start = pd.Timestamp("2017-01-01")
            train = pd.DataFrame([
                {"series_id": sid, "date": start + pd.Timedelta(days=day),
                 "store_nbr": store, "family": family,
                 "sales_log1p": float(day + sid * 100),
                 "onpromotion": 0.0, "is_val": int(day >= 50)}
                for sid, store, family in [(0, 1, "A"), (1, 2, "B")]
                for day in range(66)
            ])
            train.to_parquet(proc / "train_clean.parquet", index=False)
            (proc / "val_dates.json").write_text(json.dumps([
                (start + pd.Timedelta(days=day)).strftime("%Y-%m-%d")
                for day in range(50, 66)
            ]), encoding="utf-8")

            # The third series has no training history, as in T02's all-zero case.
            test = pd.DataFrame([
                {"id": sid * 16 + day - 66, "series_id": sid,
                 "date": start + pd.Timedelta(days=day),
                 "store_nbr": store, "family": family,
                 "onpromotion": 999.0}
                for sid, store, family in [(0, 1, "A"), (2, 1, "B"), (1, 2, "B")]
                for day in range(66, 82)
            ]).sort_values(["series_id", "date"]).reset_index(drop=True)
            test[["id", "date", "store_nbr", "family", "onpromotion"]].to_csv(
                raw / "test.csv", index=False
            )

            old_raw, old_proc = gru.RAW, gru.PROC
            try:
                gru.RAW, gru.PROC = raw, proc
                train_loader, val_loader, test_loader, _, dynamic_cols = (
                    gru.build_dataloaders(batch_size=8)
                )
            finally:
                gru.RAW, gru.PROC = old_raw, old_proc

            self.assertEqual(len(train_loader.dataset), 10)
            self.assertEqual(len(val_loader.dataset), 2)
            self.assertEqual(len(test_loader.dataset), 3)
            self.assertTrue(all(
                sample["y_target"].max() < 50 + sample["store_idx"] * 100
                for sample in train_loader.dataset.samples
            ))
            promo_col = dynamic_cols.index("onpromotion")
            samples = test_loader.dataset.samples
            self.assertEqual(samples[0]["x_dynamic"][-1, 0], 65)
            self.assertEqual(samples[2]["x_dynamic"][-1, 0], 165)
            self.assertTrue(np.all(samples[1]["x_dynamic"] == 0))
            self.assertTrue(all(np.all(s["x_dynamic"][:, promo_col] == 0) for s in samples))

            def make_features(frame, is_test):
                out = frame[["series_id", "date", "store_nbr", "family"]].copy()
                for col in FEATURES:
                    if col not in out:
                        out[col] = 0.0
                out["city"], out["state"], out["store_type"] = "C", "S", "T"
                out["cluster"] = out["store_nbr"]
                out["dayofweek"] = out["date"].dt.day_of_week
                if is_test:
                    out["id"] = frame["id"].to_numpy()
                    out["is_zero_series"] = (out["series_id"] == 2).astype("int8")
                else:
                    out[TARGET] = frame[TARGET].to_numpy()
                    out["is_val"] = frame["is_val"].to_numpy()
                return out.sort_values(["series_id", "date"]).reset_index(drop=True)

            make_features(train, False).to_parquet(proc / "train_fe.parquet", index=False)
            make_features(test, True).to_parquet(proc / "test_fe.parquet", index=False)
            result = run_xgboost(proc, root / "predictions", n_estimators=8,
                                 early_stopping_rounds=3, n_jobs=2)
            val_pred = np.load(result["val_path"])
            test_pred = np.load(result["test_path"])
            self.assertEqual(val_pred.shape, (32,))
            self.assertEqual(test_pred.shape, (48,))
            self.assertTrue(np.all(test_pred[32:48] == 0))
            self.assertTrue(result["model_path"].is_file())


if __name__ == "__main__":
    unittest.main()
