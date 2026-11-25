
import pandas as pd

from src.backtesting import recursive_backtest
from src.config import MONTHLY_DEMAND_FILE, RESULTS_DIR
from src.forecasting import (
    FEATURE_COLUMNS,
    TARGET_COLUMN,
    build_forecasting_pipeline,
    calculate_metrics,
    create_lag_features,
)


FORECAST_HORIZON = 6

FORECAST_ORIGINS = [
    "2016-12",
    "2017-03",
    "2017-06",
]

MODELS = {
    "Random Forest Recursive": "model_prediction",
    "Fixed-Origin Naive Baseline": "baseline_prediction",
}


def calculate_comparison_metrics(
    predictions_df: pd.DataFrame,
    group_by_origin: bool = True,
) -> pd.DataFrame:
    """Calculate forecast metrics for each model and origin."""

    rows = []

    if group_by_origin:
        groups = predictions_df.groupby("origin")
    else:
        groups = [("all_origins", predictions_df)]

    for origin, group in groups:
        actual = group["actual_quantity"]

        for model_name, prediction_column in MODELS.items():
            metrics = calculate_metrics(
                actual,
                group[prediction_column],
            )

            rows.append(
                {
                    "origin": origin,
                    "model_name": model_name,
                    "forecast_horizon": FORECAST_HORIZON,
                    "test_rows": len(group),
                    **metrics,
                }
            )

    return pd.DataFrame(rows)


def run_rolling_origin_backtest():
    """Train and evaluate six-month forecasts at three cutoffs."""

    if not MONTHLY_DEMAND_FILE.exists():
        raise FileNotFoundError(
            "Monthly demand file not found. "
            "Run 'python main.py' first."
        )

    monthly_df = pd.read_csv(
        MONTHLY_DEMAND_FILE,
        parse_dates=["order_year_month_date"],
    )

    lagged_df = create_lag_features(monthly_df)

    categories = monthly_df["category"].nunique()
    prediction_frames = []

    for origin in FORECAST_ORIGINS:
        cutoff = pd.Timestamp(f"{origin}-01")

        test_end = cutoff + pd.DateOffset(
            months=FORECAST_HORIZON
        )

        dates = lagged_df["order_year_month_date"]

        # Train only on months up to the forecast origin.
        train_df = lagged_df.loc[
            dates <= cutoff
        ].copy()

        # Evaluate on the six subsequent months.
        test_df = lagged_df.loc[
            (dates > cutoff) & (dates <= test_end)
        ].copy()

        expected_rows = categories * FORECAST_HORIZON

        if len(test_df) != expected_rows:
            raise ValueError(
                f"Origin {origin}: expected "
                f"{expected_rows} test rows, "
                f"found {len(test_df)}."
            )

        if train_df.empty:
            raise ValueError(
                f"Origin {origin}: empty training set."
            )

        # Fit a fresh model at each origin.
        model = build_forecasting_pipeline()

        model.fit(
            train_df[FEATURE_COLUMNS],
            train_df[TARGET_COLUMN],
        )

        # Never use actual future demand to update lags.
        predictions = recursive_backtest(
            model_pipeline=model,
            train_df=train_df,
            test_df=test_df,
        )

        predictions.insert(
            0,
            "origin",
            origin,
        )

        prediction_frames.append(predictions)

        print(
            f"Completed origin {origin}: "
            f"{len(train_df)} training rows, "
            f"{len(predictions)} forecast rows."
        )

    all_predictions = pd.concat(
        prediction_frames,
        ignore_index=True,
    )

    # Metrics for each forecast origin.
    origin_metrics = calculate_comparison_metrics(
        all_predictions,
        group_by_origin=True,
    )

    # Pooled metrics across all origins.
    overall_metrics = calculate_comparison_metrics(
        all_predictions,
        group_by_origin=False,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save predictions.
    all_predictions.to_csv(
        RESULTS_DIR / "rolling_origin_predictions.csv",
        index=False,
    )

    # Save metrics by origin.
    origin_metrics.to_csv(
        RESULTS_DIR / "rolling_origin_metrics.csv",
        index=False,
    )

    # Save pooled metrics.
    overall_metrics.to_csv(
        RESULTS_DIR / "rolling_origin_summary.csv",
        index=False,
    )

    print("\nRolling-Origin Metrics by Origin:")
    print(
        origin_metrics.round(4).to_string(
            index=False
        )
    )

    print("\nPooled Rolling-Origin Summary:")
    print(
        overall_metrics.round(4).to_string(
            index=False
        )
    )

    print(
        "\nNote: WAPE and MAPE are ratios "
        "(0.25 represents 25%)."
    )

    print(
        "Forecast windows overlap; pooled rows "
        "are not independent calendar-month observations."
    )

    print(
        "\nRolling-origin reports saved successfully."
    )

    return (
        all_predictions,
        origin_metrics,
        overall_metrics,
    )


if __name__ == "__main__":
    run_rolling_origin_backtest()
