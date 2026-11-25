
"""
Inventory recommendations based on demand forecasts.

Without actual stock inputs, recommendations are
illustrative scenarios, not operational decisions.
"""

from collections.abc import Mapping
from math import ceil, floor, isfinite, sqrt

import numpy as np
import pandas as pd

from src.config import INVENTORY_RECOMMENDATIONS_FILE


# =====================================================
# INVENTORY RECOMMENDATIONS
# =====================================================

def calculate_inventory_recommendations(
    monthly_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    lead_time_months: float = 1.0,
    service_level_z: float = 1.65,
    current_stock_by_category: Mapping[str, float] | None = None,
    scenario_stock_multiplier: float = 1.10,
) -> pd.DataFrame:
    """
    Calculate category-level inventory recommendations.

    If actual stock is unavailable, create an
    explicitly labeled illustrative scenario.

    Safety stock assumes independent monthly
    demand variability. Historical demand
    standard deviation is used as a proxy.

    Fractional lead times are supported by
    prorating forecast demand in the final month.
    """

    # -------------------------------------------------
    # VALIDATE PARAMETERS
    # -------------------------------------------------

    parameters = [
        ("lead_time_months", lead_time_months, True),
        ("service_level_z", service_level_z, False),
        (
            "scenario_stock_multiplier",
            scenario_stock_multiplier,
            True,
        ),
    ]

    for name, value, must_be_positive in parameters:

        if (
            not isinstance(value, (int, float))
            or not isfinite(value)
        ):
            raise ValueError(
                f"{name} must be a finite number."
            )

        invalid = (
            value <= 0
            if must_be_positive
            else value < 0
        )

        if invalid:
            raise ValueError(
                f"{name} is outside its allowed range."
            )

    # -------------------------------------------------
    # VALIDATE DATAFRAMES
    # -------------------------------------------------

    required_monthly = {
        "category",
        "total_quantity",
        "order_year_month",
        "order_year_month_date",
    }

    required_forecast = {
        "category",
        "forecast_step",
        "predicted_quantity",
    }

    for name, dataframe, required in [
        ("monthly_df", monthly_df, required_monthly),
        ("forecast_df", forecast_df, required_forecast),
    ]:

        missing = required - set(dataframe.columns)

        if missing:
            raise ValueError(
                f"{name} is missing columns: {sorted(missing)}"
            )

        if dataframe.empty:
            raise ValueError(
                f"{name} must not be empty."
            )

    history = monthly_df.copy()
    forecasts = forecast_df.copy()

    history["order_year_month_date"] = pd.to_datetime(
        history["order_year_month_date"]
    )

    history["total_quantity"] = pd.to_numeric(
        history["total_quantity"],
        errors="raise",
    )

    forecasts["predicted_quantity"] = pd.to_numeric(
        forecasts["predicted_quantity"],
        errors="raise",
    )

    forecasts["forecast_step"] = pd.to_numeric(
        forecasts["forecast_step"],
        errors="raise",
    )

    if (
        not np.isfinite(history["total_quantity"]).all()
        or (history["total_quantity"] < 0).any()
    ):
        raise ValueError(
            "Historical demand must be finite and non-negative."
        )

    if (
        not np.isfinite(forecasts["predicted_quantity"]).all()
        or (forecasts["predicted_quantity"] < 0).any()
    ):
        raise ValueError(
            "Forecast quantities must be finite and non-negative."
        )

    if (
        not np.isfinite(forecasts["forecast_step"]).all()
        or (forecasts["forecast_step"] % 1 != 0).any()
    ):
        raise ValueError(
            "Forecast steps must be positive integers."
        )

    forecasts["forecast_step"] = (
        forecasts["forecast_step"].astype(int)
    )

    if (forecasts["forecast_step"] < 1).any():
        raise ValueError(
            "Forecast steps must start at 1."
        )

    categories = set(
        history["category"].unique()
    )

    if categories != set(
        forecasts["category"].unique()
    ):
        raise ValueError(
            "Historical and forecast categories must match."
        )

    # -------------------------------------------------
    # HISTORICAL DEMAND STATISTICS
    # -------------------------------------------------

    historical_stats = (
        history.groupby(
            "category",
            as_index=False,
        )
        .agg(
            avg_monthly_demand=(
                "total_quantity",
                "mean",
            ),
            demand_std=(
                "total_quantity",
                "std",
            ),
            last_observed_month=(
                "order_year_month_date",
                "max",
            ),
        )
    )

    historical_stats["last_observed_month"] = (
        historical_stats["last_observed_month"]
        .dt.strftime("%Y-%m")
    )

    # -------------------------------------------------
    # LAST 3 MONTHS AVERAGE
    # -------------------------------------------------

    last_3_month_avg = (
        history
        .sort_values(
            ["category", "order_year_month_date"]
        )
        .groupby(
            "category",
            group_keys=False,
        )
        .tail(3)
        .groupby(
            "category",
            as_index=False,
        )
        .agg(
            last_3_month_avg_demand=(
                "total_quantity",
                "mean",
            )
        )
    )

    # -------------------------------------------------
    # FORECAST DEMAND DURING LEAD TIME
    # -------------------------------------------------

    full_months = floor(lead_time_months)

    fractional_month = (
        lead_time_months - full_months
    )

    needed_months = max(
        6,
        ceil(lead_time_months),
    )

    forecast_summary_rows = []

    for category, group in forecasts.groupby(
        "category"
    ):

        group = group.sort_values(
            "forecast_step"
        )

        steps = group[
            "forecast_step"
        ].tolist()

        expected_steps = list(
            range(1, len(group) + 1)
        )

        if steps != expected_steps:
            raise ValueError(
                f"Missing or duplicate forecast "
                f"steps for {category}."
            )

        if len(group) < needed_months:
            raise ValueError(
                f"Not enough forecast months "
                f"for {category}. "
                f"At least {needed_months} are required."
            )

        future = group[
            "predicted_quantity"
        ].to_numpy(dtype=float)

        # Sum forecast demand during full lead-time months.
        lead_time_demand = float(
            future[:full_months].sum()
        )

        # Add fractional month demand if applicable.
        if fractional_month > 0:
            lead_time_demand += float(
                fractional_month
                * future[full_months]
            )

        forecast_summary_rows.append(
            {
                "category": category,
                "next_month_forecast": float(
                    future[0]
                ),
                "avg_forecast_next_6_months": float(
                    future[:6].mean()
                ),
                "max_forecast_next_6_months": float(
                    future[:6].max()
                ),
                "forecast_demand_during_lead_time": (
                    lead_time_demand
                ),
            }
        )

    forecast_summary = pd.DataFrame(
        forecast_summary_rows
    )

    # -------------------------------------------------
    # MERGE DEMAND AND FORECAST INFORMATION
    # -------------------------------------------------

    inventory_df = historical_stats.merge(
        last_3_month_avg,
        on="category",
        how="inner",
        validate="one_to_one",
    ).merge(
        forecast_summary,
        on="category",
        how="inner",
        validate="one_to_one",
    )

    inventory_df["demand_std"] = (
        inventory_df["demand_std"].fillna(0)
    )

    # -------------------------------------------------
    # CURRENT STOCK: ACTUAL OR ILLUSTRATIVE
    # -------------------------------------------------

    if current_stock_by_category is None:

        inventory_df["estimated_current_stock"] = (
            inventory_df["last_3_month_avg_demand"]
            * scenario_stock_multiplier
        )

        inventory_df["stock_source"] = (
            "Illustrative scenario (not observed)"
        )

        print(
            "\nWARNING: Actual inventory data was not supplied."
        )

        print(
            "Stock levels, order quantities, and risk "
            "labels are illustrative scenario outputs only."
        )

    else:

        if not isinstance(
            current_stock_by_category,
            Mapping,
        ):
            raise ValueError(
                "current_stock_by_category must be a mapping."
            )

        if set(current_stock_by_category) != categories:
            raise ValueError(
                "Provide actual stock for every category."
            )

        try:
            stock = {
                category: float(value)
                for category, value
                in current_stock_by_category.items()
            }

        except (TypeError, ValueError) as error:
            raise ValueError(
                "Stock values must be numeric."
            ) from error

        if any(
            not isfinite(value) or value < 0
            for value in stock.values()
        ):
            raise ValueError(
                "Stock must be finite and non-negative."
            )

        # Retain the original column for
        # compatibility with existing reports.
        inventory_df["estimated_current_stock"] = (
            inventory_df["category"].map(stock)
        )

        inventory_df["stock_source"] = (
            "Provided current stock"
        )

    # -------------------------------------------------
    # SAFETY STOCK
    # -------------------------------------------------

    inventory_df["safety_stock"] = (
        service_level_z
        * inventory_df["demand_std"]
        * sqrt(lead_time_months)
    )

    # -------------------------------------------------
    # REORDER POINT
    # -------------------------------------------------

    inventory_df["reorder_point"] = (
        inventory_df["forecast_demand_during_lead_time"]
        + inventory_df["safety_stock"]
    )

    # -------------------------------------------------
    # RECOMMENDED ORDER QUANTITY
    # -------------------------------------------------

    inventory_df["recommended_order_quantity"] = (
        np.ceil(
            np.maximum(
                inventory_df["reorder_point"]
                - inventory_df["estimated_current_stock"],
                0,
            )
        )
    )

    # -------------------------------------------------
    # HEURISTIC STOCKOUT RISK
    # -------------------------------------------------

    inventory_df["stockout_risk"] = np.where(
        inventory_df["estimated_current_stock"]
        < inventory_df["forecast_demand_during_lead_time"],
        "High",
        np.where(
            inventory_df["estimated_current_stock"]
            < inventory_df["reorder_point"],
            "Medium",
            "Low",
        ),
    )

    if current_stock_by_category is None:

        inventory_df["risk_interpretation"] = (
            "Heuristic scenario label"
        )

    else:

        inventory_df["risk_interpretation"] = (
            "Heuristic label; not a calibrated probability"
        )

    inventory_df["lead_time_months"] = (
        lead_time_months
    )

    # -------------------------------------------------
    # ROUND NUMERIC OUTPUTS
    # -------------------------------------------------

    numeric_columns = [
        "avg_monthly_demand",
        "demand_std",
        "last_3_month_avg_demand",
        "next_month_forecast",
        "avg_forecast_next_6_months",
        "max_forecast_next_6_months",
        "forecast_demand_during_lead_time",
        "estimated_current_stock",
        "safety_stock",
        "reorder_point",
        "recommended_order_quantity",
    ]

    inventory_df[numeric_columns] = (
        inventory_df[numeric_columns].round(2)
    )

    # -------------------------------------------------
    # SAVE RESULTS
    # -------------------------------------------------

    INVENTORY_RECOMMENDATIONS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    inventory_df.to_csv(
        INVENTORY_RECOMMENDATIONS_FILE,
        index=False,
    )

    print(
        "Inventory recommendations saved to: "
        f"{INVENTORY_RECOMMENDATIONS_FILE}"
    )

    return inventory_df
