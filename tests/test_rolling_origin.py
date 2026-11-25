
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MONTHLY_FILE = (
    ROOT / "data" / "processed"
    / "monthly_category_demand.csv"
)


class TestRollingOrigin(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.predictions = pd.read_csv(
            RESULTS / "rolling_origin_predictions.csv"
        )

        cls.origin_metrics = pd.read_csv(
            RESULTS / "rolling_origin_metrics.csv"
        )

        cls.summary = pd.read_csv(
            RESULTS / "rolling_origin_summary.csv"
        )

        cls.monthly = pd.read_csv(
            MONTHLY_FILE,
            parse_dates=["order_year_month_date"],
        )

    def test_forecast_windows(self):
        """Verify three complete six-month forecast windows."""

        df = self.predictions

        expected_origins = [
            "2016-12",
            "2017-03",
            "2017-06",
        ]

        self.assertEqual(len(df), 54)

        self.assertEqual(
            sorted(df["origin"].unique()),
            expected_origins,
        )

        for (origin, category), group in df.groupby(
            ["origin", "category"]
        ):
            with self.subTest(
                origin=origin,
                category=category,
            ):
                group = group.sort_values(
                    "forecast_step"
                )

                self.assertEqual(len(group), 6)

                self.assertEqual(
                    group["forecast_step"].tolist(),
                    [1, 2, 3, 4, 5, 6],
                )

                expected_months = [
                    str(
                        pd.Period(origin, freq="M") + step
                    )
                    for step in range(1, 7)
                ]

                self.assertEqual(
                    group["forecast_month"].tolist(),
                    expected_months,
                )

        self.assertTrue(
            np.isfinite(df["model_prediction"]).all()
        )

        self.assertTrue(
            (df["model_prediction"] >= 0).all()
        )

    def test_fixed_origin_baseline(self):
        """Baseline must use the last known training demand."""

        for (origin, category), group in (
            self.predictions.groupby(
                ["origin", "category"]
            )
        ):
            history = self.monthly.loc[
                (
                    self.monthly["category"]
                    == category
                )
                & (
                    self.monthly[
                        "order_year_month_date"
                    ]
                    <= pd.Timestamp(f"{origin}-01")
                )
            ].sort_values(
                "order_year_month_date"
            )

            self.assertFalse(history.empty)

            last_actual = float(
                history["total_quantity"].iloc[-1]
            )

            np.testing.assert_allclose(
                group["baseline_prediction"],
                last_actual,
            )

    def test_saved_metrics(self):
        """Recalculate saved metrics independently."""

        comparisons = [
            (
                self.predictions.groupby("origin"),
                self.origin_metrics,
            ),
            (
                [("all_origins", self.predictions)],
                self.summary,
            ),
        ]

        model_columns = {
            "Random Forest Recursive": (
                "model_prediction"
            ),
            "Fixed-Origin Naive Baseline": (
                "baseline_prediction"
            ),
        }

        for groups, metrics_table in comparisons:
            for origin, group in groups:
                actual = group[
                    "actual_quantity"
                ].to_numpy(dtype=float)

                for model_name, column in (
                    model_columns.items()
                ):
                    predicted = group[
                        column
                    ].to_numpy(dtype=float)

                    errors = actual - predicted

                    rmse = np.sqrt(
                        np.mean(errors ** 2)
                    )

                    mae = np.mean(
                        np.abs(errors)
                    )

                    wape = (
                        np.sum(np.abs(errors))
                        / np.sum(np.abs(actual))
                    )

                    saved = metrics_table.loc[
                        (
                            metrics_table["origin"]
                            == origin
                        )
                        & (
                            metrics_table["model_name"]
                            == model_name
                        )
                    ]

                    self.assertEqual(len(saved), 1)

                    saved = saved.iloc[0]

                    self.assertEqual(
                        int(saved["test_rows"]),
                        len(group),
                    )

                    for metric, expected in {
                        "rmse": rmse,
                        "mae": mae,
                        "wape": wape,
                    }.items():
                        with self.subTest(
                            origin=origin,
                            model=model_name,
                            metric=metric,
                        ):
                            self.assertAlmostEqual(
                                float(saved[metric]),
                                float(expected),
                                places=6,
                            )


if __name__ == "__main__":
    unittest.main()
