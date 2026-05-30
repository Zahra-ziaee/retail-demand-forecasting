import pandas as pd

from src.config import RESULTS_DIR
from src.data_loader import load_raw_superstore_data, print_data_summary
from src.forecasting import (
    create_lag_features,
    forecast_future_demand,
    save_forecasting_model,
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


def main():
    print_section("Retail Demand Forecasting & Inventory Optimization")

    raw_df = load_raw_superstore_data()

    print_section("Raw Data")
    print_data_summary(raw_df)

    print_section("Data Cleaning")
    cleaned_df = clean_superstore_data(raw_df)
    print(f"Cleaned data shape: {cleaned_df.shape}")

    print_section("Monthly Category Demand")
    monthly_demand_df = create_monthly_category_demand(cleaned_df)
    print(monthly_demand_df.head())
    save_processed_outputs(cleaned_df, monthly_demand_df)

    print_section("Forecasting Model")
    lagged_df = create_lag_features(monthly_demand_df)
    model_pipeline, metrics_df = train_forecasting_model(lagged_df)
    save_forecasting_model(model_pipeline)

    print_section("Future Demand Forecast")
    forecast_df = forecast_future_demand(
        model_pipeline=model_pipeline,
        monthly_df=monthly_demand_df,
    )
    print(forecast_df.head(10))

    print_section("Inventory Recommendations")
    inventory_df = calculate_inventory_recommendations(
        monthly_df=monthly_demand_df,
        forecast_df=forecast_df,
    )
    print(inventory_df)

    print_section("Visualizations")

    forecast_vs_actual_file = RESULTS_DIR / "forecast_vs_actual.csv"
    forecast_vs_actual_df = pd.read_csv(forecast_vs_actual_file)

    plot_historical_monthly_demand(monthly_demand_df)
    plot_forecasted_demand(forecast_df)
    plot_inventory_risk(inventory_df)
    plot_forecast_vs_actual(forecast_vs_actual_df)


if __name__ == "__main__":
    main()