
import matplotlib.pyplot as plt
import pandas as pd

from src.config import FIGURES_DIR


# =====================================================
# HISTORICAL MONTHLY DEMAND
# =====================================================

def plot_historical_monthly_demand(
    monthly_df: pd.DataFrame,
) -> None:

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    pivot_df = monthly_df.pivot(
        index="order_year_month",
        columns="category",
        values="total_quantity",
    )

    fig, ax = plt.subplots(figsize=(14, 6))

    for column in pivot_df.columns:
        ax.plot(
            pivot_df.index,
            pivot_df[column],
            marker="o",
            label=column,
        )

    ax.set_title("Historical Monthly Demand by Category")
    ax.set_xlabel("Month")
    ax.set_ylabel("Quantity Sold")
    ax.tick_params(axis="x", rotation=90)
    ax.legend()

    fig.tight_layout()

    output_path = FIGURES_DIR / "historical_monthly_demand.png"

    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved chart: {output_path}")


# =====================================================
# FORECASTED DEMAND
# =====================================================

def plot_forecasted_demand(
    forecast_df: pd.DataFrame,
) -> None:

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    pivot_df = forecast_df.pivot(
        index="forecast_month",
        columns="category",
        values="predicted_quantity",
    )

    fig, ax = plt.subplots(figsize=(10, 6))

    for column in pivot_df.columns:
        ax.plot(
            pivot_df.index,
            pivot_df[column],
            marker="o",
            label=column,
        )

    ax.set_title("Forecasted Demand by Category")
    ax.set_xlabel("Forecast Month")
    ax.set_ylabel("Predicted Quantity")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()

    fig.tight_layout()

    output_path = FIGURES_DIR / "forecasted_demand.png"

    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    print(f"Saved chart: {output_path}")


# =====================================================
# INVENTORY RISK - HEURISTIC SCENARIO
# =====================================================

def plot_inventory_risk(
    inventory_df: pd.DataFrame,
) -> None:
    """
    Plot heuristic inventory risk labels.

    Labels are not calibrated probabilities.
    If stock values are illustrative, clearly
    communicate this limitation in the chart.
    """

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if inventory_df.empty:
        raise ValueError(
            "Inventory dataframe must not be empty."
        )

    if "stockout_risk" not in inventory_df.columns:
        raise ValueError(
            "Missing required column: stockout_risk"
        )

    # -------------------------------------------------
    # IDENTIFY STOCK SOURCE
    # -------------------------------------------------

    if "stock_source" in inventory_df.columns:

        sources = inventory_df["stock_source"]

        if sources.eq(
            "Illustrative scenario (not observed)"
        ).all():

            chart_title = (
                "Illustrative Inventory Risk Scenario"
            )

            note = (
                "Stock quantities are assumed, not observed.\n"
                "Risk labels are heuristic, not calibrated "
                "stockout probabilities."
            )

        elif sources.eq(
            "Provided current stock"
        ).all():

            chart_title = (
                "Heuristic Inventory Risk - Provided Stock"
            )

            note = (
                "Stock quantities were provided as inputs.\n"
                "Risk labels are heuristic, not calibrated "
                "stockout probabilities."
            )

        else:

            chart_title = (
                "Heuristic Inventory Risk - Mixed Stock Sources"
            )

            note = (
                "Stock inputs may include illustrative values.\n"
                "Risk labels are heuristic, not calibrated "
                "stockout probabilities."
            )

    else:

        chart_title = (
            "Heuristic Inventory Risk - Source Unspecified"
        )

        note = (
            "Stock input source is not specified.\n"
            "Risk labels are heuristic, not calibrated "
            "stockout probabilities."
        )

    # -------------------------------------------------
    # COUNT RISK LABELS
    # -------------------------------------------------

    risk_order = [
        "High",
        "Medium",
        "Low",
    ]

    risk_counts = (
        inventory_df["stockout_risk"]
        .value_counts()
        .reindex(risk_order, fill_value=0)
    )

    # -------------------------------------------------
    # CREATE CHART
    # -------------------------------------------------

    fig, ax = plt.subplots(figsize=(9, 6))

    bars = ax.bar(
        risk_counts.index,
        risk_counts.values,
    )

    ax.set_title(
        chart_title,
        fontsize=13,
        fontweight="bold",
        pad=15,
    )

    ax.set_xlabel("Heuristic Risk Label")
    ax.set_ylabel("Number of Categories")

    ax.set_ylim(
        0,
        max(1, int(risk_counts.max())) + 1,
    )

    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.25,
    )

    ax.set_axisbelow(True)

    # Display counts above bars.
    for bar in bars:

        height = bar.get_height()

        ax.annotate(
            f"{int(height)}",
            xy=(
                bar.get_x() + bar.get_width() / 2,
                height,
            ),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )

    # -------------------------------------------------
    # ADD EXPLANATORY NOTE
    # -------------------------------------------------

    fig.text(
        0.5,
        0.05,
        note,
        ha="center",
        va="bottom",
        fontsize=9,
    )

    fig.subplots_adjust(
        left=0.12,
        right=0.95,
        top=0.88,
        bottom=0.25,
    )

    # -------------------------------------------------
    # SAVE CHART
    # -------------------------------------------------

    output_path = (
        FIGURES_DIR / "stockout_risk_distribution.png"
    )

    fig.savefig(
        output_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(f"Saved chart: {output_path}")


# =====================================================
# FORECAST VS ACTUAL
# =====================================================

def plot_forecast_vs_actual(
    forecast_vs_actual_df: pd.DataFrame,
) -> None:

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    for category, category_df in (
        forecast_vs_actual_df.groupby("category")
    ):

        category_df = category_df.sort_values(
            "order_year_month"
        )

        fig, ax = plt.subplots(figsize=(10, 5))

        ax.plot(
            category_df["order_year_month"],
            category_df["actual_quantity"],
            marker="o",
            label="Actual Demand",
        )

        ax.plot(
            category_df["order_year_month"],
            category_df["model_prediction"],
            marker="o",
            label="Model Forecast",
        )

        ax.plot(
            category_df["order_year_month"],
            category_df["baseline_prediction"],
            marker="o",
            label="Naive Baseline",
        )

        ax.set_title(
            f"Forecast vs Actual - {category}"
        )

        ax.set_xlabel("Month")
        ax.set_ylabel("Quantity")
        ax.tick_params(axis="x", rotation=45)
        ax.legend()

        fig.tight_layout()

        safe_category_name = (
            category.lower().replace(" ", "_")
        )

        output_path = (
            FIGURES_DIR
            / f"forecast_vs_actual_{safe_category_name}.png"
        )

        fig.savefig(output_path, dpi=300)
        plt.close(fig)

        print(f"Saved chart: {output_path}")
