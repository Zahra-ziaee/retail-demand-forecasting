import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIGURES_DIR


def plot_historical_monthly_demand(monthly_df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    pivot_df = monthly_df.pivot(
        index="order_year_month",
        columns="category",
        values="total_quantity",
    )

    plt.figure(figsize=(14, 6))

    for column in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[column], marker="o", label=column)

    plt.title("Historical Monthly Demand by Category")
    plt.xlabel("Month")
    plt.ylabel("Quantity Sold")
    plt.xticks(rotation=90)
    plt.legend()
    plt.tight_layout()

    output_path = FIGURES_DIR / "historical_monthly_demand.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved chart: {output_path}")


def plot_forecasted_demand(forecast_df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    pivot_df = forecast_df.pivot(
        index="forecast_month",
        columns="category",
        values="predicted_quantity",
    )

    plt.figure(figsize=(10, 6))

    for column in pivot_df.columns:
        plt.plot(pivot_df.index, pivot_df[column], marker="o", label=column)

    plt.title("Forecasted Demand by Category")
    plt.xlabel("Forecast Month")
    plt.ylabel("Predicted Quantity")
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()

    output_path = FIGURES_DIR / "forecasted_demand.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved chart: {output_path}")


def plot_inventory_risk(inventory_df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    risk_counts = inventory_df["stockout_risk"].value_counts()

    plt.figure(figsize=(8, 5))
    plt.bar(risk_counts.index, risk_counts.values)
    plt.title("Stockout Risk Distribution")
    plt.xlabel("Risk Level")
    plt.ylabel("Number of Categories")
    plt.tight_layout()

    output_path = FIGURES_DIR / "stockout_risk_distribution.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved chart: {output_path}")


def plot_forecast_vs_actual(forecast_vs_actual_df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for category, category_df in forecast_vs_actual_df.groupby("category"):
        category_df = category_df.sort_values("order_year_month")

        plt.figure(figsize=(10, 5))

        plt.plot(
            category_df["order_year_month"],
            category_df["actual_quantity"],
            marker="o",
            label="Actual Demand",
        )

        plt.plot(
            category_df["order_year_month"],
            category_df["model_prediction"],
            marker="o",
            label="Model Forecast",
        )

        plt.plot(
            category_df["order_year_month"],
            category_df["baseline_prediction"],
            marker="o",
            label="Naive Baseline",
        )

        plt.title(f"Forecast vs Actual - {category}")
        plt.xlabel("Month")
        plt.ylabel("Quantity")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()

        safe_category_name = category.lower().replace(" ", "_")
        output_path = FIGURES_DIR / f"forecast_vs_actual_{safe_category_name}.png"

        plt.savefig(output_path, dpi=300)
        plt.close()

        print(f"Saved chart: {output_path}")