import pandas as pd

from src.config import (
    MONTHLY_DEMAND_FILE,
    PROCESSED_SUPERSTORE_FILE,
)


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.columns = (
        df.columns
        .str.replace("ï»¿", "", regex=False)
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )

    return df


def clean_superstore_data(df: pd.DataFrame) -> pd.DataFrame:
    df = clean_column_names(df)

    date_columns = ["order_date", "ship_date"]

    for column in date_columns:
        if column in df.columns:
            df[column] = pd.to_datetime(
                df[column],
                errors="coerce",
                dayfirst=True,
            )

    numeric_columns = [
        "sales",
        "quantity",
        "discount",
        "profit",
        "postal_code",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.drop_duplicates()
    df = df.dropna(subset=["order_date", "quantity", "sales"])

    df["order_year"] = df["order_date"].dt.year
    df["order_month"] = df["order_date"].dt.month
    df["order_year_month"] = df["order_date"].dt.to_period("M").astype(str)

    df["profit_margin"] = df.apply(
        lambda row: row["profit"] / row["sales"] if row["sales"] != 0 else 0,
        axis=1,
    )

    return df


def create_monthly_category_demand(df: pd.DataFrame) -> pd.DataFrame:
    monthly_df = (
        df.groupby(["category", "order_year_month"], as_index=False)
        .agg(
            total_quantity=("quantity", "sum"),
            total_sales=("sales", "sum"),
            total_profit=("profit", "sum"),
            order_count=("order_id", "nunique"),
        )
    )

    monthly_df["order_year_month_date"] = pd.to_datetime(
        monthly_df["order_year_month"] + "-01"
    )
    monthly_df["year"] = monthly_df["order_year_month_date"].dt.year
    monthly_df["month"] = monthly_df["order_year_month_date"].dt.month

    monthly_df = monthly_df.sort_values(
        ["category", "order_year_month_date"]
    ).reset_index(drop=True)

    return monthly_df


def save_processed_outputs(
    cleaned_df: pd.DataFrame,
    monthly_demand_df: pd.DataFrame,
) -> None:
    PROCESSED_SUPERSTORE_FILE.parent.mkdir(parents=True, exist_ok=True)

    cleaned_df.to_csv(PROCESSED_SUPERSTORE_FILE, index=False)
    monthly_demand_df.to_csv(MONTHLY_DEMAND_FILE, index=False)

    print(f"Cleaned data saved to: {PROCESSED_SUPERSTORE_FILE}")
    print(f"Monthly demand data saved to: {MONTHLY_DEMAND_FILE}")