
import unittest

import numpy as np
import pandas as pd

from src.backtesting import recursive_backtest


class DummyForecastModel:
    """
    A deterministic model for testing.

    Prediction = lag_1 + 10
    """

    def predict(self, X):
        return X["lag_1"].to_numpy(dtype=float) + 10


class TestRecursiveBacktesting(unittest.TestCase):

    def setUp(self):

        self.model = DummyForecastModel()

        self.train_df = pd.DataFrame({
            "category": ["Furniture"] * 3,
            "order_year_month_date": pd.to_datetime([
                "2020-01-01",
                "2020-02-01",
                "2020-03-01",
            ]),
            "total_quantity": [10, 20, 30],
        })

        self.test_df = pd.DataFrame({
            "category": ["Furniture"] * 3,
            "order_year_month_date": pd.to_datetime([
                "2020-04-01",
                "2020-05-01",
                "2020-06-01",
            ]),
            "total_quantity": [999, 888, 777],
        })

    def test_recursive_predictions(self):

        result = recursive_backtest(
            model_pipeline=self.model,
            train_df=self.train_df,
            test_df=self.test_df,
        )

        # April: 30 + 10 = 40
        # May:   40 + 10 = 50
        # June:  50 + 10 = 60

        np.testing.assert_allclose(
            result["model_prediction"],
            [40, 50, 60],
        )

        self.assertEqual(
            result["forecast_step"].tolist(),
            [1, 2, 3],
        )

        self.assertEqual(
            result["forecast_month"].tolist(),
            ["2020-04", "2020-05", "2020-06"],
        )

        # Fixed-origin baseline:
        # Repeat the final observed training value.
        np.testing.assert_allclose(
            result["baseline_prediction"],
            [30, 30, 30],
        )

    def test_no_holdout_target_leakage(self):

        original = recursive_backtest(
            self.model,
            self.train_df,
            self.test_df,
        )

        modified_test = self.test_df.copy()

        modified_test["total_quantity"] = [
            5000,
            6000,
            7000,
        ]

        modified = recursive_backtest(
            self.model,
            self.train_df,
            modified_test,
        )

        # Predictions must not change when
        # actual holdout values are modified.
        np.testing.assert_allclose(
            original["model_prediction"],
            modified["model_prediction"],
        )

        # Actual values should still be recorded
        # correctly for evaluation.
        np.testing.assert_allclose(
            modified["actual_quantity"],
            [5000, 6000, 7000],
        )

    def test_non_consecutive_holdout_months(self):

        invalid_test = self.test_df.iloc[
            [0, 2]
        ].copy()

        with self.assertRaises(ValueError):
            recursive_backtest(
                self.model,
                self.train_df,
                invalid_test,
            )


if __name__ == "__main__":
    unittest.main()
