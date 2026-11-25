
import unittest

import pandas as pd

from src.forecasting import create_lag_features


class TestLagFeatures(unittest.TestCase):

    def setUp(self):
        dates = pd.date_range(
            "2020-01-01",
            periods=5,
            freq="MS",
        )

        rows = []

        for category, values in {
            "A": [10, 20, 30, 40, 50],
            "B": [100, 200, 300, 400, 500],
        }.items():

            for date, quantity in zip(dates, values):
                rows.append(
                    {
                        "category": category,
                        "order_year_month_date": date,
                        "total_quantity": quantity,
                    }
                )

        # Shuffle input to test chronological sorting.
        self.monthly_df = (
            pd.DataFrame(rows)
            .sample(frac=1, random_state=42)
            .reset_index(drop=True)
        )

    def get_row(self, df, category, month):
        selected = df.loc[
            (df["category"] == category)
            & (
                df["order_year_month_date"]
                == pd.Timestamp(month)
            )
        ]

        self.assertEqual(len(selected), 1)
        return selected.iloc[0]

    def test_lag_and_rolling_values(self):
        result = create_lag_features(self.monthly_df)

        # First 3 months in each category are excluded.
        self.assertEqual(len(result), 4)

        expected = {
            ("A", "2020-04-01"): (30, 20, 10, 20),
            ("A", "2020-05-01"): (40, 30, 20, 30),
            ("B", "2020-04-01"): (300, 200, 100, 200),
            ("B", "2020-05-01"): (400, 300, 200, 300),
        }

        for (category, month), values in expected.items():
            with self.subTest(category=category, month=month):
                row = self.get_row(result, category, month)

                actual = (
                    row["lag_1"],
                    row["lag_2"],
                    row["lag_3"],
                    row["rolling_mean_3"],
                )

                self.assertEqual(actual, values)

    def test_no_current_month_target_leakage(self):
        original = create_lag_features(self.monthly_df)

        changed = self.monthly_df.copy()

        mask = (
            (changed["category"] == "A")
            & (
                changed["order_year_month_date"]
                == pd.Timestamp("2020-04-01")
            )
        )

        changed.loc[mask, "total_quantity"] = 9999

        updated = create_lag_features(changed)

        feature_columns = [
            "lag_1",
            "lag_2",
            "lag_3",
            "rolling_mean_3",
        ]

        before = self.get_row(
            original, "A", "2020-04-01"
        )

        after = self.get_row(
            updated, "A", "2020-04-01"
        )

        # April's features must not use April's target.
        self.assertEqual(
            before[feature_columns].tolist(),
            after[feature_columns].tolist(),
        )

        # May is allowed to use April's observed demand.
        may_after = self.get_row(
            updated, "A", "2020-05-01"
        )

        self.assertEqual(may_after["lag_1"], 9999)


if __name__ == "__main__":
    unittest.main()
