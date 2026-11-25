
import pandas as pd

from src.backtesting import recursive_backtest
from src.config import RESULTS_DIR
from src.data_loader import (
    load_raw_superstore_data,
    print_data_summary,
)
from src.forecasting import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_forecasting_pipeline,
    calculate_metrics,
    create_lag_features,
    forecast_future_demand,
    save_forecasting_model,
    split_time_series_train_test,
    train_forecasting_model,
)
from src.inventory import calculate_inventory_recommendations
from src.preprocessing import (
    clean_superstore_data,
    create_monthly_category_demand,
    save_processed_outputs,
)
from src.utils import print_section
from src.visualization import (
    plot_forecast_vs_actual,
    plot_forecasted_demand,
    plot_historical_monthly_demand,
    plot_inventory_risk,
)


# =====================================================
# RECURSIVE BACKTEST EVALUATION
# =====================================================

def evaluate_recursive_holdout(
    lagged_df,
    model_pipeline,
):
    """
    Evaluate a previously fitted model using
    fixed-origin recursive forecasting.

    The model uses only training history and
    its own predictions for subsequent months.
    """

    # Reproduce the same chronological split
    # used during the original model evaluation.
    train_df, test_df = split_time_series_train_test(
        lagged_df
    )

    backtest_df = recursive_backtest(
        model_pipeline=model_pipeline,
        train_df=train_df,
        test_df=test_df,
    )

    if backtest_df.empty:
        raise ValueError(
            "Recursive backtest returned no predictions."
        )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # -------------------------------------------------
    # SAVE RECURSIVE PREDICTIONS
    # -------------------------------------------------

    predictions_file = (
        RESULTS_DIR / "recursive_backtest_predictions.csv"
    )

    backtest_df.to_csv(
        predictions_file,
        index=False,
    )

    # -------------------------------------------------
    # OVERALL RECURSIVE METRICS
    # -------------------------------------------------

    actual = backtest_df["actual_quantity"]

    model_columns = {
        "Random Forest Recursive": "model_prediction",
        "Fixed-Origin Naive Baseline": "baseline_prediction",
    }

    metrics_rows = []

    for model_name, prediction_column in model_columns.items():

        predictions = backtest_df[
            prediction_column
        ]

        metrics = calculate_metrics(
            actual,
            predictions,
        )

        metrics_rows.append(
            {
                "model_name": model_name,
                "evaluation_type": "fixed_origin_recursive",
                "forecast_horizon_months": int(
                    backtest_df["forecast_step"].max()
                ),
                "test_rows": len(backtest_df),
                **metrics,
            }
        )

    recursive_metrics_df = pd.DataFrame(
        metrics_rows
    )

    metrics_file = (
        RESULTS_DIR / "recursive_backtest_metrics.csv"
    )

    recursive_metrics_df.to_csv(
        metrics_file,
        index=False,
    )

    print("\nRecursive Backtesting - Overall Metrics")

    print(
        recursive_metrics_df.round(4).to_string(
            index=False
        )
    )

    # -------------------------------------------------
    # CATEGORY-LEVEL RECURSIVE METRICS
    # -------------------------------------------------

    category_rows = []

    for category, category_df in backtest_df.groupby(
        "category"
    ):

        for model_name, prediction_column in model_columns.items():

            metrics = calculate_metrics(
                category_df["actual_quantity"],
                category_df[prediction_column],
            )

            category_rows.append(
                {
                    "category": category,
                    "model_name": model_name,
                    "test_rows": len(category_df),
                    **metrics,
                }
            )

    category_metrics_df = pd.DataFrame(
        category_rows
    )

    category_file = (
        RESULTS_DIR
        / "recursive_backtest_category_metrics.csv"
    )

    category_metrics_df.to_csv(
        category_file,
        index=False,
    )

    print("\nRecursive Backtesting - Category Metrics")

    print(
        category_metrics_df.round(4).to_string(
            index=False
        )
    )

    print(
        "\nNote: WAPE and MAPE are stored as ratios. "
        "For example, 0.25 represents 25%."
    )

    print("\nSaved recursive backtesting reports:")
    print(predictions_file)
    print(metrics_file)
    print(category_file)

    return backtest_df, recursive_metrics_df


# =====================================================
# MAIN PIPELINE
# =====================================================

def main():

    print_section(
        "Retail Demand Forecasting & Inventory Optimization"
    )

    # -------------------------------------------------
    # 1. LOAD RAW DATA
    # -------------------------------------------------

    print_section("Raw Data")

    raw_df = load_raw_superstore_data()

    print_data_summary(
        raw_df
    )

    # -------------------------------------------------
    # 2. DATA CLEANING
    # -------------------------------------------------

    print_section("Data Cleaning")

    cleaned_df = clean_superstore_data(
        raw_df
    )

    print(
        f"Cleaned data shape: {cleaned_df.shape}"
    )

    # -------------------------------------------------
    # 3. MONTHLY CATEGORY DEMAND
    # -------------------------------------------------

    print_section("Monthly Category Demand")

    monthly_demand_df = (
        create_monthly_category_demand(
            cleaned_df
        )
    )

    print(
        monthly_demand_df.head()
    )

    save_processed_outputs(
        cleaned_df,
        monthly_demand_df,
    )

    # -------------------------------------------------
    # 4. FEATURE ENGINEERING
    # -------------------------------------------------

    print_section("Lag Feature Engineering")

    lagged_df = create_lag_features(
        monthly_demand_df
    )

    print(
        f"Lagged dataset shape: {lagged_df.shape}"
    )

    # -------------------------------------------------
    # 5. TRAIN AND EVALUATE ON HOLDOUT
    # -------------------------------------------------

    print_section(
        "Forecasting Model - One-Step Evaluation"
    )

    # This model is trained only on the training split.
    # It is used for evaluation, not final deployment.

    evaluation_model_pipeline, one_step_metrics_df = (
        train_forecasting_model(
            lagged_df
        )
    )

    # -------------------------------------------------
    # 6. RECURSIVE BACKTESTING
    # -------------------------------------------------

    print_section(
        "Recursive Backtesting - Multi-Month Evaluation"
    )

    (
        recursive_predictions_df,
        recursive_metrics_df,
    ) = evaluate_recursive_holdout(
        lagged_df=lagged_df,
        model_pipeline=evaluation_model_pipeline,
    )

    print(
        "\nRecursive predictions:",
        len(recursive_predictions_df),
    )

    # -------------------------------------------------
    # 7. FINAL MODEL TRAINING
    # -------------------------------------------------

    print_section(
        "Final Model Training - Full Historical Data"
    )

    # After holdout evaluation is complete,
    # fit a fresh model on all available history.
    #
    # The final model is separate from the
    # model used to calculate evaluation metrics.

    final_model_pipeline = build_forecasting_pipeline()

    X_full = lagged_df[
        FEATURE_COLUMNS
    ]

    y_full = lagged_df[
        TARGET_COLUMN
    ]

    print(
        f"Training final model on {len(X_full)} historical rows..."
    )

    final_model_pipeline.fit(
        X_full,
        y_full,
    )

    print(
        "Final model training completed successfully."
    )

    # Save only the final, full-history model.
    save_forecasting_model(
        final_model_pipeline
    )

    # -------------------------------------------------
    # 8. FUTURE DEMAND FORECASTING
    # -------------------------------------------------

    print_section(
        "Future Demand Forecast - Final Model"
    )

    # IMPORTANT:
    # Forecast the future using the final model,
    # not the holdout evaluation model.

    forecast_df = forecast_future_demand(
        model_pipeline=final_model_pipeline,
        monthly_df=monthly_demand_df,
    )

    print(
        forecast_df.head(10)
    )

    # -------------------------------------------------
    # 9. INVENTORY RECOMMENDATIONS
    # -------------------------------------------------

    print_section(
        "Inventory Recommendations"
    )

    inventory_df = (
        calculate_inventory_recommendations(
            monthly_df=monthly_demand_df,
            forecast_df=forecast_df,
        )
    )

    print(
        inventory_df
    )

    # -------------------------------------------------
    # 10. VISUALIZATIONS
    # -------------------------------------------------

    print_section("Visualizations")

    forecast_vs_actual_file = (
        RESULTS_DIR / "forecast_vs_actual.csv"
    )

    forecast_vs_actual_df = pd.read_csv(
        forecast_vs_actual_file
    )

    plot_historical_monthly_demand(
        monthly_demand_df
    )

    plot_forecasted_demand(
        forecast_df
    )

    plot_inventory_risk(
        inventory_df
    )

    plot_forecast_vs_actual(
        forecast_vs_actual_df
    )

    # -------------------------------------------------
    # COMPLETED
    # -------------------------------------------------

    print_section(
        "Pipeline Completed Successfully"
    )


if __name__ == "__main__":
    main()
