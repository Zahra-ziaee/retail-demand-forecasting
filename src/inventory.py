import numpy as np
import pandas as pd

from src.config import INVENTORY_RECOMMENDATIONS_FILE


def calculate_inventory_recommendations(
    monthly_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    lead_time_months: float = 1.0,
    service_level_z: float = 1.65,
) -> pd.DataFrame:
    historical_stats = (
        monthly_df.groupby("category", as_index=False)
        .agg(
            avg_monthly_demand=("total_quantity", "mean"),
            demand_std=("total_quantity", "std"),
            last_observed_month=("order_year_month", "max"),
        )
    )

    last_3_month_avg = (
        monthly_df.sort_values("order_year_month_date")
        .groupby("category")
        .tail(3)
        .groupby("category", as_index=False)
        .agg(last_3_month_avg_demand=("total_quantity", "mean"))
    )

    forecast_summary = (
        forecast_df.groupby("category", as_index=False)
        .agg(
            next_month_forecast=("predicted_quantity", "first"),
            avg_forecast_next_6_months=("predicted_quantity", "mean"),
            max_forecast_next_6_months=("predicted_quantity", "max"),
        )
    )

    inventory_df = historical_stats.merge(
        last_3_month_avg,
        on="category",
        how="left",
    ).merge(
        forecast_summary,
        on="category",
        how="left",
    )

    inventory_df["demand_std"] = inventory_df["demand_std"].fillna(0)

    inventory_df["estimated_current_stock"] = (
        inventory_df["last_3_month_avg_demand"] * 1.10
    )

    inventory_df["safety_stock"] = (
        service_level_z
        * inventory_df["demand_std"]
        * np.sqrt(lead_time_months)
    )

    inventory_df["reorder_point"] = (
        inventory_df["avg_forecast_next_6_months"] * lead_time_months
        + inventory_df["safety_stock"]
    )

    inventory_df["recommended_order_quantity"] = (
        inventory_df["reorder_point"] - inventory_df["estimated_current_stock"]
    ).clip(lower=0)

    inventory_df["stockout_risk"] = np.where(
        inventory_df["estimated_current_stock"] < inventory_df["next_month_forecast"],
        "High",
        np.where(
            inventory_df["estimated_current_stock"] < inventory_df["reorder_point"],
            "Medium",
            "Low",
        ),
    )

    numeric_columns = [
        "avg_monthly_demand",
        "demand_std",
        "last_3_month_avg_demand",
        "next_month_forecast",
        "avg_forecast_next_6_months",
        "max_forecast_next_6_months",
        "estimated_current_stock",
        "safety_stock",
        "reorder_point",
        "recommended_order_quantity",
    ]

    for column in numeric_columns:
        inventory_df[column] = inventory_df[column].round(2)

    INVENTORY_RECOMMENDATIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    inventory_df.to_csv(INVENTORY_RECOMMENDATIONS_FILE, index=False)

    print(f"Inventory recommendations saved to: {INVENTORY_RECOMMENDATIONS_FILE}")

    return inventory_df