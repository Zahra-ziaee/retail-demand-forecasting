
import tempfile
import unittest
from math import ceil, sqrt
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.inventory import calculate_inventory_recommendations


class TestInventoryRecommendations(unittest.TestCase):

    def setUp(self):
        # Save test outputs in a temporary directory.
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)

        self.output_file = (
            Path(self.tempdir.name)
            / "inventory_recommendations.csv"
        )

        patcher = patch(
            "src.inventory.INVENTORY_RECOMMENDATIONS_FILE",
            self.output_file,
        )
        patcher.start()
        self.addCleanup(patcher.stop)

        self.monthly_df = pd.DataFrame({
            "category": ["Furniture"] * 3,
            "order_year_month": [
                "2020-01",
                "2020-02",
                "2020-03",
            ],
            "order_year_month_date": pd.to_datetime([
                "2020-01-01",
                "2020-02-01",
                "2020-03-01",
            ]),
            "total_quantity": [10, 20, 30],
        })

        self.forecast_df = pd.DataFrame({
            "category": ["Furniture"] * 6,
            "forecast_step": [1, 2, 3, 4, 5, 6],
            "predicted_quantity": [
                50, 60, 70, 80, 90, 100
            ],
        })

    def calculate(self, **kwargs):
        result = calculate_inventory_recommendations(
            monthly_df=self.monthly_df,
            forecast_df=self.forecast_df,
            **kwargs,
        )

        self.assertTrue(self.output_file.exists())
        self.assertEqual(len(result), 1)

        return result.iloc[0]

    def test_illustrative_stock_scenario(self):
        row = self.calculate(
            lead_time_months=1,
            service_level_z=1,
        )

        # Historical demand: [10, 20, 30]
        # Average = 20
        # Sample standard deviation = 10
        # Illustrative stock = 20 * 1.10 = 22
        # Lead-time demand = 50
        # Safety stock = 10
        # Reorder point = 60
        # Order quantity = 60 - 22 = 38

        self.assertAlmostEqual(
            row["estimated_current_stock"], 22
        )

        self.assertAlmostEqual(
            row["forecast_demand_during_lead_time"], 50
        )

        self.assertAlmostEqual(
            row["safety_stock"], 10
        )

        self.assertAlmostEqual(
            row["reorder_point"], 60
        )

        self.assertEqual(
            row["recommended_order_quantity"], 38
        )

        self.assertEqual(
            row["stock_source"],
            "Illustrative scenario (not observed)",
        )

        self.assertEqual(
            row["risk_interpretation"],
            "Heuristic scenario label",
        )

    def test_provided_stock_risk_levels(self):
        scenarios = [
            (20, "High"),
            (55, "Medium"),
            (90, "Low"),
        ]

        for stock, expected_risk in scenarios:
            with self.subTest(stock=stock):
                row = self.calculate(
                    lead_time_months=1,
                    service_level_z=1,
                    current_stock_by_category={
                        "Furniture": stock
                    },
                )

                self.assertEqual(
                    row["stockout_risk"],
                    expected_risk,
                )

                self.assertEqual(
                    row["stock_source"],
                    "Provided current stock",
                )

                expected_order = max(
                    ceil(60 - stock),
                    0,
                )

                self.assertEqual(
                    row["recommended_order_quantity"],
                    expected_order,
                )

    def test_fractional_and_two_month_lead_time(self):
        scenarios = [
            (1.5, 80),
            (2.0, 110),
        ]

        for lead_time, expected_demand in scenarios:
            with self.subTest(lead_time=lead_time):
                row = self.calculate(
                    lead_time_months=lead_time,
                    service_level_z=1,
                    current_stock_by_category={
                        "Furniture": 85
                    },
                )

                expected_safety = 10 * sqrt(
                    lead_time
                )

                expected_reorder = (
                    expected_demand + expected_safety
                )

                expected_order = max(
                    ceil(expected_reorder - 85),
                    0,
                )

                self.assertAlmostEqual(
                    row["forecast_demand_during_lead_time"],
                    expected_demand,
                    places=2,
                )

                self.assertAlmostEqual(
                    row["reorder_point"],
                    expected_reorder,
                    places=2,
                )

                self.assertEqual(
                    row["recommended_order_quantity"],
                    expected_order,
                )

    def test_invalid_inputs(self):
        with self.assertRaises(ValueError):
            self.calculate(
                lead_time_months=0,
            )

        with self.assertRaises(ValueError):
            self.calculate(
                current_stock_by_category={
                    "Furniture": -5
                },
            )

        with self.assertRaises(ValueError):
            self.calculate(
                lead_time_months=7,
            )


if __name__ == "__main__":
    unittest.main()
