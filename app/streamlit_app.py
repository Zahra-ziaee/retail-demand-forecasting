
from pathlib import Path

import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RESULTS_DIR = PROJECT_ROOT / "results"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_outputs():

    required_files = {
        "monthly_demand": (
            PROCESSED_DIR / "monthly_category_demand.csv"
        ),
        "forecast": (
            RESULTS_DIR / "demand_forecast.csv"
        ),
        "inventory": (
            RESULTS_DIR / "inventory_recommendations.csv"
        ),
        "metrics": (
            RESULTS_DIR / "forecast_metrics.csv"
        ),
        "category_metrics": (
            RESULTS_DIR / "category_forecast_metrics.csv"
        ),
        "forecast_vs_actual": (
            RESULTS_DIR / "forecast_vs_actual.csv"
        ),
    }

    data = {}

    for name, path in required_files.items():
        data[name] = pd.read_csv(path)

    optional_files = {
        "recursive_metrics": (
            RESULTS_DIR / "recursive_backtest_metrics.csv"
        ),
        "rolling_metrics": (
            RESULTS_DIR / "rolling_origin_metrics.csv"
        ),
        "rolling_summary": (
            RESULTS_DIR / "rolling_origin_summary.csv"
        ),
    }

    for name, path in optional_files.items():

        if path.exists():
            data[name] = pd.read_csv(path)
        else:
            data[name] = pd.DataFrame()

    return data


# =====================================================
# FORMAT HELPERS
# =====================================================

def format_number(value):
    return f"{float(value):,.2f}"


def format_percent(value):
    return f"{float(value) * 100:.2f}%"


def format_metrics_table(df):

    display = df.copy()

    for column in ["wape", "mape"]:

        if column in display.columns:
            display[column] = (
                display[column].apply(format_percent)
            )

    for column in ["rmse", "mae"]:

        if column in display.columns:
            display[column] = (
                display[column].round(2)
            )

    if "r2" in display.columns:
        display["r2"] = display["r2"].round(3)

    return display


# =====================================================
# MAIN DASHBOARD
# =====================================================

def main():

    st.set_page_config(
        page_title="Retail Demand Forecasting",
        layout="wide",
    )

    st.title(
        "Retail Demand Forecasting & Inventory Planning"
    )

    st.markdown(
        """
        **Historical demand forecasting and inventory
        planning demonstration**

        This dashboard evaluates category-level demand
        forecasts and presents illustrative inventory
        planning scenarios using Superstore sales data.
        """
    )

    st.info(
        "Historical simulation: the dataset ends in "
        "December 2017. Forecasts for January-June 2018 "
        "are not current or live demand forecasts."
    )

    if st.sidebar.button("Reload CSV reports"):
        st.cache_data.clear()
        st.rerun()

    try:
        data = load_outputs()

    except (FileNotFoundError, pd.errors.EmptyDataError) as error:

        st.error(
            "Required data or report files are missing "
            "or empty."
        )

        st.code(
            "python main.py\n"
            "python -m src.rolling_origin"
        )

        st.caption(str(error))
        return

    monthly_demand = data["monthly_demand"]
    forecast = data["forecast"]
    inventory = data["inventory"]
    metrics = data["metrics"]
    category_metrics = data["category_metrics"]
    forecast_vs_actual = data["forecast_vs_actual"]

    recursive_metrics = data["recursive_metrics"]
    rolling_metrics = data["rolling_metrics"]
    rolling_summary = data["rolling_summary"]

    if (
        monthly_demand.empty
        or forecast.empty
        or inventory.empty
        or metrics.empty
    ):
        st.error("One or more required reports are empty.")
        return

    # =================================================
    # DATASET OVERVIEW
    # =================================================

    st.divider()

    st.subheader("Dataset Overview")

    categories = sorted(
        monthly_demand["category"].unique()
    )

    historical_start = (
        monthly_demand["order_year_month"].min()
    )

    historical_end = (
        monthly_demand["order_year_month"].max()
    )

    forecast_start = (
        forecast["forecast_month"].min()
    )

    forecast_end = (
        forecast["forecast_month"].max()
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Product Categories",
        len(categories),
    )

    col2.metric(
        "Historical End",
        str(historical_end),
    )

    col3.metric(
        "Forecast Horizon",
        f"{forecast['forecast_step'].nunique()} months",
    )

    st.caption(
        f"Historical data: {historical_start} to "
        f"{historical_end}. "
        f"Forecast simulation: {forecast_start} to "
        f"{forecast_end}."
    )

    # =================================================
    # ONE-STEP MODEL EVALUATION
    # =================================================

    st.divider()

    st.subheader("One-Step Forecast Evaluation")

    st.caption(
        "This evaluation uses actual observed demand "
        "from preceding months. It is not the same as "
        "a six-month recursive forecast."
    )

    model_rows = metrics.loc[
        metrics["model_name"]
        == "Random Forest Forecasting Model"
    ]

    if model_rows.empty:

        st.warning(
            "Random Forest evaluation metrics "
            "were not found."
        )

    else:

        model_metric = model_rows.iloc[0]

        col1, col2, col3, col4, col5 = (
            st.columns(5)
        )

        col1.metric(
            "RMSE",
            format_number(model_metric["rmse"]),
        )

        col2.metric(
            "MAE",
            format_number(model_metric["mae"]),
        )

        col3.metric(
            "WAPE",
            format_percent(model_metric["wape"]),
        )

        col4.metric(
            "MAPE",
            format_percent(model_metric["mape"]),
        )

        col5.metric(
            "R2",
            f"{model_metric['r2']:.3f}",
        )

    st.markdown("#### Random Forest vs Naive Baseline")

    st.dataframe(
        format_metrics_table(metrics),
        width="stretch",
        hide_index=True,
    )

    # =================================================
    # RECURSIVE BACKTESTING
    # =================================================

    st.divider()

    st.subheader("Six-Month Recursive Backtesting")

    st.caption(
        "The model predicts six future months from "
        "a fixed historical origin. Future lag features "
        "are updated using earlier predictions, not "
        "the hidden actual demand."
    )

    if recursive_metrics.empty:

        st.warning(
            "Recursive backtesting results not found. "
            "Run python main.py."
        )

    else:

        st.dataframe(
            format_metrics_table(
                recursive_metrics
            ),
            width="stretch",
            hide_index=True,
        )

    # =================================================
    # ROLLING-ORIGIN EVALUATION
    # =================================================

    st.divider()

    st.subheader(
        "Expanding-Window Rolling-Origin Evaluation"
    )

    st.caption(
        "Three historical training cutoffs: "
        "December 2016, March 2017, and June 2017. "
        "A new model is fitted at each cutoff, "
        "then evaluated over six future months."
    )

    if (
        rolling_metrics.empty
        or rolling_summary.empty
    ):

        st.warning(
            "Rolling-origin reports are unavailable. "
            "Run: python -m src.rolling_origin"
        )

    else:

        st.markdown("#### Metrics by Forecast Origin")

        st.dataframe(
            format_metrics_table(
                rolling_metrics
            ),
            width="stretch",
            hide_index=True,
        )

        st.markdown("#### Pooled Results")

        st.dataframe(
            format_metrics_table(
                rolling_summary
            ),
            width="stretch",
            hide_index=True,
        )

        rf_rows = rolling_summary.loc[
            rolling_summary["model_name"]
            == "Random Forest Recursive"
        ]

        baseline_rows = rolling_summary.loc[
            rolling_summary["model_name"]
            == "Fixed-Origin Naive Baseline"
        ]

        if (
            not rf_rows.empty
            and not baseline_rows.empty
        ):

            rf = rf_rows.iloc[0]
            baseline = baseline_rows.iloc[0]

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Recursive RF WAPE",
                format_percent(rf["wape"]),
            )

            col2.metric(
                "Naive Baseline WAPE",
                format_percent(baseline["wape"]),
            )

            col3.metric(
                "Evaluated Forecast Rows",
                int(rf["test_rows"]),
            )

        st.info(
            "Rolling-origin forecast windows overlap. "
            "The pooled forecast rows are not "
            "independent calendar-month observations. "
            "This is not an independent-sample "
            "statistical significance test."
        )

    # =================================================
    # CATEGORY SELECTION
    # =================================================

    st.divider()

    st.subheader("Category Demand Analysis")

    selected_category = st.selectbox(
        "Select Product Category",
        categories,
    )

    category_history = monthly_demand.loc[
        monthly_demand["category"]
        == selected_category
    ].copy()

    category_forecast = forecast.loc[
        forecast["category"]
        == selected_category
    ].copy()

    category_actual = forecast_vs_actual.loc[
        forecast_vs_actual["category"]
        == selected_category
    ].copy()

    # =================================================
    # HISTORICAL AND FORECAST CHARTS
    # =================================================

    col_left, col_right = st.columns(2)

    with col_left:

        st.markdown("#### Historical Monthly Demand")

        history_chart = (
            category_history[
                ["order_year_month", "total_quantity"]
            ]
            .sort_values("order_year_month")
            .set_index("order_year_month")
        )

        st.line_chart(history_chart)

    with col_right:

        st.markdown(
            "#### Forecast Simulation: Jan-Jun 2018"
        )

        forecast_chart = (
            category_forecast[
                ["forecast_month", "predicted_quantity"]
            ]
            .sort_values("forecast_month")
            .set_index("forecast_month")
        )

        st.line_chart(forecast_chart)

    # =================================================
    # FORECAST VS ACTUAL
    # =================================================

    st.divider()

    st.subheader(
        "Forecast vs Actual - One-Step Holdout"
    )

    st.caption(
        "These predictions belong to the one-step "
        "evaluation, not the rolling-origin "
        "recursive evaluation."
    )

    category_actual = category_actual.sort_values(
        "order_year_month"
    )

    comparison_chart = category_actual[
        [
            "order_year_month",
            "actual_quantity",
            "model_prediction",
            "baseline_prediction",
        ]
    ].set_index("order_year_month")

    st.line_chart(comparison_chart)

    st.markdown("#### Category-Level Metrics")

    selected_metrics = category_metrics.loc[
        category_metrics["category"]
        == selected_category
    ].copy()

    st.dataframe(
        format_metrics_table(selected_metrics),
        width="stretch",
        hide_index=True,
    )

    # =================================================
    # INVENTORY PLANNING
    # =================================================

    st.divider()

    st.subheader(
        "Inventory Planning - Assumption-Based Scenario"
    )

    selected_inventory_rows = inventory.loc[
        inventory["category"]
        == selected_category
    ]

    if selected_inventory_rows.empty:

        st.error(
            "Inventory results for the selected "
            "category are missing."
        )
        return

    selected_inventory = (
        selected_inventory_rows.iloc[0]
    )

    stock_source = str(
        selected_inventory.get(
            "stock_source",
            "Source unspecified",
        )
    )

    illustrative = (
        stock_source
        == "Illustrative scenario (not observed)"
    )

    if illustrative:

        st.warning(
            "Actual warehouse stock was not provided. "
            "Stock quantities are estimated using "
            "a historical-demand scenario. "
            "Suggested orders and risk labels "
            "are illustrative, not operational decisions."
        )

    else:

        st.info(
            f"Stock input source: {stock_source}. "
            "Reorder recommendations still depend "
            "on forecasting and inventory-policy "
            "assumptions."
        )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Next-Month Demand Forecast",
        format_number(
            selected_inventory["next_month_forecast"]
        ),
    )

    col2.metric(
        "Safety Stock",
        format_number(
            selected_inventory["safety_stock"]
        ),
    )

    col3.metric(
        "Reorder Point",
        format_number(
            selected_inventory["reorder_point"]
        ),
    )

    col4.metric(
        "Heuristic Risk Label",
        str(selected_inventory["stockout_risk"]),
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Assumed / Supplied Stock",
        format_number(
            selected_inventory["estimated_current_stock"]
        ),
    )

    col2.metric(
        "Demand During Lead Time",
        format_number(
            selected_inventory[
                "forecast_demand_during_lead_time"
            ]
        ),
    )

    col3.metric(
        "Scenario Order Quantity"
        if illustrative
        else "Calculated Order Quantity",
        format_number(
            selected_inventory[
                "recommended_order_quantity"
            ]
        ),
    )

    st.caption(
        "Risk is a rule-based High/Medium/Low label, "
        "not a calibrated probability of stockout. "
        "Safety-stock calculations assume a simplified "
        "demand-variability model."
    )

    st.markdown("#### Full Inventory Scenario Table")

    st.dataframe(
        inventory,
        width="stretch",
        hide_index=True,
    )

    # =================================================
    # FULL FORECAST TABLE
    # =================================================

    st.divider()

    st.subheader(
        "Historical Future-Forecast Simulation"
    )

    st.caption(
        "Six monthly forecasts generated from "
        "the December 2017 historical endpoint."
    )

    st.dataframe(
        forecast,
        width="stretch",
        hide_index=True,
    )

    # =================================================
    # BUSINESS INTERPRETATION
    # =================================================

    st.divider()

    st.subheader("Business Insights and Limitations")

    st.markdown(
        """
        - The Random Forest is compared with simple
          naive baselines, rather than reported alone.
        - Recursive validation avoids using future
          actual demand as lag inputs.
        - Rolling-origin evaluation checks performance
          across multiple historical cutoffs.
        - Category-level forecasting errors can vary
          substantially and should be reviewed separately.
        - Inventory calculations illustrate how forecasts
          can inform reorder-point decisions.
        - Without actual warehouse stock, supplier
          information, and inventory costs, the suggested
          order quantities are not deployable
          purchasing recommendations.
        - The dataset ends in 2017; this dashboard
          does not represent live retail demand.
        """
    )


if __name__ == "__main__":
    main()
