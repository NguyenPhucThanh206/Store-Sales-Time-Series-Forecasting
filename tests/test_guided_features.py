import unittest

import numpy as np
import pandas as pd

from src.features.guided_features import GUIDED_FEATURES, add_guided_features, restrict_promo_leads


class GuidedFeatureTests(unittest.TestCase):
    def test_masked_horizon_and_available_promotions(self):
        dates = pd.date_range("2017-01-01", periods=100)
        frames = []
        for sid in (0, 1):
            frames.append(pd.DataFrame({
                "series_id": sid, "store_nbr": 1, "family": f"F{sid}", "city": "C",
                "date": dates, "sales_log1p": np.log1p(np.arange(100) + sid),
                "onpromotion": np.arange(100, dtype="float32"),
            }))
        frame = pd.concat(frames, ignore_index=True)
        transactions = pd.DataFrame({"date": dates, "store_nbr": 1,
                                     "transactions": np.arange(100)})
        baseline = add_guided_features(frame.copy(), transactions)
        selected = frame["date"].between(dates[70], dates[85])
        masked = frame.copy()
        masked.loc[selected, "sales_log1p"] = np.nan
        altered = add_guided_features(masked, transactions)
        pd.testing.assert_frame_equal(baseline.loc[selected, GUIDED_FEATURES].reset_index(drop=True),
                                      altered.loc[selected, GUIDED_FEATURES].reset_index(drop=True))
        self.assertAlmostEqual(baseline.loc[70, "transactions_mean_7_lag16"], 51.0)
        limited = restrict_promo_leads(baseline.loc[selected])
        first_series = limited[limited["series_id"].eq(0)]
        self.assertEqual(first_series.iloc[-1]["promo_lead_mean_7"], 85)
        self.assertEqual(first_series.iloc[-2]["promo_lead_mean_7"], 85)


if __name__ == "__main__":
    unittest.main()
