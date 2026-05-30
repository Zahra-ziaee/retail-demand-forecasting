from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


@st.cache_data
def load_outputs():
    monthly_demand = pd.read_csv(PROCESSED_DIR / "monthly_category_demand.csv")
    forecast = pd.read_csv(RESULTS_DIR / "demand_forecast.csv")
    inventory = pd.read_csv(RESULTS_DIR / "inventory_recommendations.csv")
    metrics = pd.read_csv(RESULTS_DIR / "forecast_metrics.csv")

    return monthly_demand, forecast, inventory, metrics


def format_number(value):
    return f"{value:,.2f}"


def main():
    st.set_page_config(
        page_title="Retail Demand Forecasting Dashboard",
        layout="wide",
    )

    st.title("📦 Retail Demand Forecasting & Inventory Optimization")

    st.write(
        "This dashboard forecasts future product-category demand and provides "
        "inventory recommendations such as safety stock, reorder point, "
        "recommended order quantity, and stockout risk."
    )

    try:
        monthly_demand, forecast, inventory, metrics = load_outputs()

    except FileNotFoundError:
        st.error("Required output files were not found.")
        st.write("Please run the pipeline first:")
        st.code("python main.py")
        return

    st.divider()

    st.subheader("Forecast Model Performance")

    metric_row = metrics.iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("RMSE", format_number(metric_row["rmse"]))
    col2.metric("MAE", format_number(metric_row["mae"]))
    col3.metric("R² Score", f"{metric_row['r2']:.3f}")
    col4.metric("Test Rows", int(metric_row["test_rows"]))

    st.divider()

    st.subheader("Demand Overview")

    selected_category = st.selectbox(
        "Select Category",
        sorted(monthly_demand["category"].unique()),
    )

    category_history = monthly_demand[
        monthly_demand["category"] == selected_category
    ].copy()

    category_forecast = forecast[
        forecast["category"] == selected_category
    ].copy()

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Historical Monthly Demand")

        history_chart = category_history[
            ["order_year_month", "total_quantity"]
        ].copy()

        history_chart = history_chart.sort_values("order_year_month")
        history_chart = history_chart.set_index("order_year_month")

        st.line_chart(history_chart)

    with col_right:
        st.markdown("### Forecasted Demand - Next 6 Months")

        forecast_chart = category_forecast[
            ["forecast_month", "predicted_quantity"]
        ].copy()

        forecast_chart = forecast_chart.sort_values("forecast_month")
        forecast_chart = forecast_chart.set_index("forecast_month")

        st.line_chart(forecast_chart)

    st.divider()

    st.subheader("Inventory Recommendations")

    inventory_display = inventory.copy()

    selected_inventory = inventory_display[
        inventory_display["category"] == selected_category
    ].iloc[0]

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Next Month Forecast",
        format_number(selected_inventory["next_month_forecast"]),
    )

    col2.metric(
        "Safety Stock",
        format_number(selected_inventory["safety_stock"]),
    )

    col3.metric(
        "Reorder Point",
        format_number(selected_inventory["reorder_point"]),
    )

    col4.metric(
        "Stockout Risk",
        selected_inventory["stockout_risk"],
    )

    st.markdown("### Full Inventory Recommendation Table")

    st.dataframe(
        inventory_display,
        width="stretch",
        hide_index=True,
    )

    st.divider()

    st.subheader("Forecast Table")

    st.dataframe(
        forecast,
        width="stretch",
        hide_index=True,
    )

    st.divider()

    st.subheader("Business Insights")

    st.markdown(
        """
        - The model forecasts category-level demand for the next six months.
        - Safety stock is calculated using historical demand variability.
        - Reorder point combines expected demand and safety stock.
        - Stockout risk helps identify whether current stock assumptions are enough for near-future demand.
        - This project moves from descriptive analytics to predictive and prescriptive analytics.
        """
    )


if __name__ == "__main__":
    main()