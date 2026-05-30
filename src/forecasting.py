from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    FORECAST_HORIZON_MONTHS,
    FORECAST_METRICS_FILE,
    FORECAST_MODEL_FILE,
    FORECAST_RESULTS_FILE,
    RANDOM_STATE,
    RESULTS_DIR,
)


FEATURE_COLUMNS = [
    "category",
    "year",
    "month",
    "lag_1",
    "lag_2",
    "lag_3",
    "rolling_mean_3",
]

TARGET_COLUMN = "total_quantity"

FORECAST_VS_ACTUAL_FILE = RESULTS_DIR / "forecast_vs_actual.csv"
CATEGORY_METRICS_FILE = RESULTS_DIR / "category_forecast_metrics.csv"


def create_lag_features(monthly_df: pd.DataFrame) -> pd.DataFrame:
    df = monthly_df.copy()
    df = df.sort_values(["category", "order_year_month_date"])

    for lag in [1, 2, 3]:
        df[f"lag_{lag}"] = df.groupby("category")["total_quantity"].shift(lag)

    df["rolling_mean_3"] = (
        df.groupby("category")["total_quantity"]
        .shift(1)
        .rolling(window=3)
        .mean()
        .reset_index(level=0, drop=True)
    )

    df = df.dropna(
        subset=["lag_1", "lag_2", "lag_3", "rolling_mean_3"]
    ).reset_index(drop=True)

    return df


def build_forecasting_pipeline() -> Pipeline:
    numeric_features = [
        "year",
        "month",
        "lag_1",
        "lag_2",
        "lag_3",
        "rolling_mean_3",
    ]

    categorical_features = ["category"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_features,
            ),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model),
        ]
    )

    return pipeline


def calculate_wape(y_true, y_pred) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    denominator = np.sum(np.abs(y_true))

    if denominator == 0:
        return np.nan

    return np.sum(np.abs(y_true - y_pred)) / denominator


def calculate_mape(y_true, y_pred) -> float:
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    non_zero_mask = y_true != 0

    if non_zero_mask.sum() == 0:
        return np.nan

    return np.mean(
        np.abs((y_true[non_zero_mask] - y_pred[non_zero_mask]) / y_true[non_zero_mask])
    )


def calculate_metrics(y_true, y_pred) -> dict:
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    wape = calculate_wape(y_true, y_pred)
    mape = calculate_mape(y_true, y_pred)

    if len(y_true) > 1:
        r2 = r2_score(y_true, y_pred)
    else:
        r2 = np.nan

    return {
        "rmse": rmse,
        "mae": mae,
        "wape": wape,
        "mape": mape,
        "r2": r2,
    }


def split_time_series_train_test(lagged_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    train_rows = []
    test_rows = []

    for category, group in lagged_df.groupby("category"):
        group = group.sort_values("order_year_month_date")
        test_size = min(6, max(1, len(group) // 5))

        train_rows.append(group.iloc[:-test_size])
        test_rows.append(group.iloc[-test_size:])

    train_df = pd.concat(train_rows, ignore_index=True)
    test_df = pd.concat(test_rows, ignore_index=True)

    return train_df, test_df


def evaluate_forecasting_model(
    test_df: pd.DataFrame,
    model_predictions: np.ndarray,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y_test = test_df[TARGET_COLUMN].values

    # Baseline: predict current month demand using previous month's demand.
    baseline_predictions = test_df["lag_1"].values

    model_metrics = calculate_metrics(y_test, model_predictions)
    baseline_metrics = calculate_metrics(y_test, baseline_predictions)

    overall_metrics_df = pd.DataFrame(
        [
            {
                "model_name": "Random Forest Forecasting Model",
                "rmse": model_metrics["rmse"],
                "mae": model_metrics["mae"],
                "wape": model_metrics["wape"],
                "mape": model_metrics["mape"],
                "r2": model_metrics["r2"],
                "test_rows": len(test_df),
            },
            {
                "model_name": "Naive Baseline Lag-1",
                "rmse": baseline_metrics["rmse"],
                "mae": baseline_metrics["mae"],
                "wape": baseline_metrics["wape"],
                "mape": baseline_metrics["mape"],
                "r2": baseline_metrics["r2"],
                "test_rows": len(test_df),
            },
        ]
    )

    forecast_vs_actual_df = test_df[
        [
            "category",
            "order_year_month",
            "order_year_month_date",
            "total_quantity",
            "lag_1",
        ]
    ].copy()

    forecast_vs_actual_df = forecast_vs_actual_df.rename(
        columns={
            "total_quantity": "actual_quantity",
            "lag_1": "baseline_prediction",
        }
    )

    forecast_vs_actual_df["model_prediction"] = model_predictions.round(2)
    forecast_vs_actual_df["baseline_prediction"] = forecast_vs_actual_df[
        "baseline_prediction"
    ].round(2)

    forecast_vs_actual_df["model_abs_error"] = (
        forecast_vs_actual_df["actual_quantity"]
        - forecast_vs_actual_df["model_prediction"]
    ).abs()

    forecast_vs_actual_df["baseline_abs_error"] = (
        forecast_vs_actual_df["actual_quantity"]
        - forecast_vs_actual_df["baseline_prediction"]
    ).abs()

    category_metric_rows = []

    for category, category_df in forecast_vs_actual_df.groupby("category"):
        y_category = category_df["actual_quantity"].values
        model_category_pred = category_df["model_prediction"].values
        baseline_category_pred = category_df["baseline_prediction"].values

        category_model_metrics = calculate_metrics(y_category, model_category_pred)
        category_baseline_metrics = calculate_metrics(y_category, baseline_category_pred)

        category_metric_rows.append(
            {
                "category": category,
                "model_name": "Random Forest Forecasting Model",
                "rmse": category_model_metrics["rmse"],
                "mae": category_model_metrics["mae"],
                "wape": category_model_metrics["wape"],
                "mape": category_model_metrics["mape"],
                "r2": category_model_metrics["r2"],
                "test_rows": len(category_df),
            }
        )

        category_metric_rows.append(
            {
                "category": category,
                "model_name": "Naive Baseline Lag-1",
                "rmse": category_baseline_metrics["rmse"],
                "mae": category_baseline_metrics["mae"],
                "wape": category_baseline_metrics["wape"],
                "mape": category_baseline_metrics["mape"],
                "r2": category_baseline_metrics["r2"],
                "test_rows": len(category_df),
            }
        )

    category_metrics_df = pd.DataFrame(category_metric_rows)

    return overall_metrics_df, category_metrics_df, forecast_vs_actual_df


def train_forecasting_model(
    lagged_df: pd.DataFrame,
) -> Tuple[Pipeline, pd.DataFrame]:
    train_df, test_df = split_time_series_train_test(lagged_df)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[FEATURE_COLUMNS]

    model_pipeline = build_forecasting_pipeline()

    print("\nTraining demand forecasting model...")
    model_pipeline.fit(X_train, y_train)
    print("Training finished.")

    model_predictions = model_pipeline.predict(X_test)

    overall_metrics_df, category_metrics_df, forecast_vs_actual_df = (
        evaluate_forecasting_model(
            test_df=test_df,
            model_predictions=model_predictions,
        )
    )

    FORECAST_METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)

    overall_metrics_df.to_csv(FORECAST_METRICS_FILE, index=False)
    category_metrics_df.to_csv(CATEGORY_METRICS_FILE, index=False)
    forecast_vs_actual_df.to_csv(FORECAST_VS_ACTUAL_FILE, index=False)

    print("\nForecast evaluation:")
    print(overall_metrics_df)

    print("\nCategory-level forecast metrics:")
    print(category_metrics_df)

    print(f"\nForecast metrics saved to: {FORECAST_METRICS_FILE}")
    print(f"Category metrics saved to: {CATEGORY_METRICS_FILE}")
    print(f"Forecast vs actual saved to: {FORECAST_VS_ACTUAL_FILE}")

    return model_pipeline, overall_metrics_df


def save_forecasting_model(model_pipeline: Pipeline) -> None:
    FORECAST_MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_pipeline, FORECAST_MODEL_FILE)

    print(f"Forecasting model saved to: {FORECAST_MODEL_FILE}")


def forecast_future_demand(
    model_pipeline: Pipeline,
    monthly_df: pd.DataFrame,
    horizon_months: int = FORECAST_HORIZON_MONTHS,
) -> pd.DataFrame:
    forecast_rows = []

    for category, group in monthly_df.groupby("category"):
        group = group.sort_values("order_year_month_date").copy()

        demand_history = group["total_quantity"].tolist()
        last_date = group["order_year_month_date"].max()

        for step in range(1, horizon_months + 1):
            future_date = last_date + pd.DateOffset(months=step)

            lag_1 = demand_history[-1]
            lag_2 = demand_history[-2]
            lag_3 = demand_history[-3]
            rolling_mean_3 = np.mean(demand_history[-3:])

            input_row = pd.DataFrame(
                [
                    {
                        "category": category,
                        "year": future_date.year,
                        "month": future_date.month,
                        "lag_1": lag_1,
                        "lag_2": lag_2,
                        "lag_3": lag_3,
                        "rolling_mean_3": rolling_mean_3,
                    }
                ]
            )

            predicted_demand = float(model_pipeline.predict(input_row)[0])
            predicted_demand = max(predicted_demand, 0)

            forecast_rows.append(
                {
                    "category": category,
                    "forecast_month": future_date.strftime("%Y-%m"),
                    "forecast_step": step,
                    "predicted_quantity": round(predicted_demand, 2),
                }
            )

            demand_history.append(predicted_demand)

    forecast_df = pd.DataFrame(forecast_rows)

    FORECAST_RESULTS_FILE.parent.mkdir(parents=True, exist_ok=True)
    forecast_df.to_csv(FORECAST_RESULTS_FILE, index=False)

    print(f"Demand forecast saved to: {FORECAST_RESULTS_FILE}")

    return forecast_df