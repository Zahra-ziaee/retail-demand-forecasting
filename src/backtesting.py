
import numpy as np
import pandas as pd

from src.forecasting import FEATURE_COLUMNS


def recursive_backtest(
    model_pipeline,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Evaluate a fitted model on a multi-month holdout.

    The model receives actual demand only up to
    the training cutoff.

    Each predicted value is recursively used to
    create features for subsequent forecast months.
    """

    required_columns = {
        "category",
        "order_year_month_date",
        "total_quantity",
    }

    for name, dataframe in [
        ("train_df", train_df),
        ("test_df", test_df),
    ]:
        missing = required_columns - set(dataframe.columns)

        if missing:
            raise ValueError(
                f"{name} is missing columns: {sorted(missing)}"
            )

    forecast_rows = []

    for category, holdout in test_df.groupby("category"):

        history_df = train_df.loc[
            train_df["category"] == category
        ].sort_values("order_year_month_date")

        holdout = holdout.sort_values(
            "order_year_month_date"
        )

        if len(history_df) < 3:
            raise ValueError(
                f"Not enough training history for {category}"
            )

        demand_history = (
            history_df["total_quantity"]
            .astype(float)
            .tolist()
        )

        last_train_month = pd.Timestamp(
            history_df["order_year_month_date"].iloc[-1]
        ).to_period("M")

        # Fixed-origin naive baseline:
        # Repeat the last observed training demand.
        baseline_prediction = demand_history[-1]

        for step, (_, row) in enumerate(
            holdout.iterrows(),
            start=1,
        ):

            forecast_date = pd.Timestamp(
                row["order_year_month_date"]
            )

            expected_month = last_train_month + step

            if forecast_date.to_period("M") != expected_month:
                raise ValueError(
                    f"Non-consecutive holdout months "
                    f"for category {category}"
                )

            # Use only the training history and
            # previously generated predictions.
            lag_1 = demand_history[-1]
            lag_2 = demand_history[-2]
            lag_3 = demand_history[-3]

            rolling_mean_3 = float(
                np.mean(demand_history[-3:])
            )

            input_row = pd.DataFrame(
                [{
                    "category": category,
                    "year": forecast_date.year,
                    "month": forecast_date.month,
                    "lag_1": lag_1,
                    "lag_2": lag_2,
                    "lag_3": lag_3,
                    "rolling_mean_3": rolling_mean_3,
                }]
            )

            predicted_demand = float(
                model_pipeline.predict(
                    input_row[FEATURE_COLUMNS]
                )[0]
            )

            predicted_demand = max(
                predicted_demand, 0.0
            )

            forecast_rows.append(
                {
                    "category": category,
                    "forecast_month": (
                        forecast_date.strftime("%Y-%m")
                    ),
                    "forecast_step": step,
                    "actual_quantity": float(
                        row["total_quantity"]
                    ),
                    "model_prediction": predicted_demand,
                    "baseline_prediction": (
                        baseline_prediction
                    ),
                }
            )

            # Crucial: append the prediction,
            # not the actual holdout demand.
            demand_history.append(
                predicted_demand
            )

    return pd.DataFrame(forecast_rows)
