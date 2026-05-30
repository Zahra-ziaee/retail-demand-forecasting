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


def train_forecasting_model(
    lagged_df: pd.DataFrame,
) -> Tuple[Pipeline, pd.DataFrame]:
    df = lagged_df.copy()

    train_rows = []
    test_rows = []

    for category, group in df.groupby("category"):
        group = group.sort_values("order_year_month_date")
        test_size = min(6, max(1, len(group) // 5))

        train_rows.append(group.iloc[:-test_size])
        test_rows.append(group.iloc[-test_size:])

    train_df = pd.concat(train_rows, ignore_index=True)
    test_df = pd.concat(test_rows, ignore_index=True)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[TARGET_COLUMN]

    model_pipeline = build_forecasting_pipeline()

    print("\nTraining demand forecasting model...")
    model_pipeline.fit(X_train, y_train)
    print("Training finished.")

    predictions = model_pipeline.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    metrics_df = pd.DataFrame(
        [
            {
                "rmse": rmse,
                "mae": mae,
                "r2": r2,
                "test_rows": len(test_df),
            }
        ]
    )

    FORECAST_METRICS_FILE.parent.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(FORECAST_METRICS_FILE, index=False)

    print("\nForecast evaluation:")
    print(metrics_df)
    print(f"Forecast metrics saved to: {FORECAST_METRICS_FILE}")

    return model_pipeline, metrics_df


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