from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"

RAW_SUPERSTORE_FILE = RAW_DATA_DIR / "superstore.csv"

PROCESSED_SUPERSTORE_FILE = PROCESSED_DATA_DIR / "superstore_cleaned.csv"
MONTHLY_DEMAND_FILE = PROCESSED_DATA_DIR / "monthly_category_demand.csv"

FORECAST_RESULTS_FILE = RESULTS_DIR / "demand_forecast.csv"
FORECAST_METRICS_FILE = RESULTS_DIR / "forecast_metrics.csv"
INVENTORY_RECOMMENDATIONS_FILE = RESULTS_DIR / "inventory_recommendations.csv"

FORECAST_MODEL_FILE = MODELS_DIR / "demand_forecasting_model.joblib"

FORECAST_HORIZON_MONTHS = 6
RANDOM_STATE = 42