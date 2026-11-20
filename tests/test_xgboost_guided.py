import unittest

import numpy as np
import pandas as pd
import xgboost as xgb

from src.models.xgboost_guided import margin, qmatrix


class MarginTests(unittest.TestCase):
    def test_margin_fallback_and_prediction(self):
        frame = pd.DataFrame({"feature": [1., 2., 3., 4.],
                              "rolling_mean_28": [2., np.nan, np.nan, np.nan],
                              "rolling_mean_7": [3., 1., np.nan, np.nan],
                              "sales_lag_16": [4., 4., 5., np.nan],
                              "sales_log1p": [2., 1., 5., 0.]})
        expected = np.array([2, 1, 5, 0], dtype="float32")
        np.testing.assert_array_equal(margin(frame), expected)
        config = {"margin": True, "groups": []}
        training = qmatrix(frame, ["feature"], config, label=True)
        evaluation = qmatrix(frame, ["feature"], config, ref=training)
        empty = xgb.train({"tree_method": "hist", "objective": "reg:squarederror"},
                          training, num_boost_round=0)
        np.testing.assert_allclose(empty.predict(evaluation), expected)


if __name__ == "__main__":
    unittest.main()
