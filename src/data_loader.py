"""Data ingestion for King County house sales dataset."""

from __future__ import annotations

import io
from pathlib import Path

import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"

# Public mirrors of Kaggle King County house sales data
DATASET_URLS = [
    "https://raw.githubusercontent.com/tthien92/KC_Housing_Data_Modeling/master/kc_house_data.csv",
    "https://raw.githubusercontent.com/ADDYBOY/Houses_Kc_Data_Prediction/master/kc_house_data.csv",
    "https://raw.githubusercontent.com/EasonSYC/kc-house-price-prediction/master/kc_house_data.csv",
]
DEFAULT_RAW_PATH = RAW_DATA_DIR / "kc_house_data.csv"
DEFAULT_PROCESSED_PATH = PROCESSED_DATA_DIR / "kc_house_clean.csv"

NUMERIC_COLUMNS = [
    "price",
    "bedrooms",
    "bathrooms",
    "sqft_living",
    "sqft_lot",
    "floors",
    "waterfront",
    "view",
    "condition",
    "grade",
    "sqft_above",
    "sqft_basement",
    "yr_built",
    "yr_renovated",
    "zipcode",
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
]

CATEGORICAL_COLUMNS = ["zipcode"]
TARGET_COLUMN = "price"
ID_COLUMNS = ["id", "date"]


def download_raw_data(dest: Path | None = None, force: bool = False) -> Path:
    """Download King County dataset if not present locally with fallback mirrors."""
    dest = dest or DEFAULT_RAW_PATH
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not force:
        return dest

    last_error = None
    for url in DATASET_URLS:
        try:
            response = requests.get(url, timeout=120)
            if response.status_code == 200 and len(response.content) > 10000:
                dest.write_bytes(response.content)
                return dest
        except Exception as exc:
            last_error = exc
            continue

    raise RuntimeError(
        f"Failed to download King County dataset from all mirror sources. Last error: {last_error}"
    )


def load_raw_data(path: Path | None = None) -> pd.DataFrame:
    """Load raw CSV, downloading if needed."""
    path = path or download_raw_data()
    df = pd.read_csv(path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply industry-standard cleaning steps."""
    data = df.copy()

    # Schema enforcement & date parsing
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
        data["sale_year"] = data["date"].dt.year.fillna(2014).astype(int)
        data["sale_month"] = data["date"].dt.month.fillna(5).astype(int)
    else:
        data["sale_year"] = 2014
        data["sale_month"] = 5

    # Numeric conversion with coercion (handles '?' or formatting issues in raw data)
    numeric_cols_to_coerce = [
        "price", "bedrooms", "bathrooms", "sqft_living", "sqft_lot",
        "floors", "waterfront", "view", "condition", "grade",
        "sqft_above", "sqft_basement", "yr_built", "yr_renovated",
        "lat", "long", "sqft_living15", "sqft_lot15"
    ]
    for col in numeric_cols_to_coerce:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    # Impute binary / ordinal features where missing means absent
    if "waterfront" in data.columns:
        data["waterfront"] = data["waterfront"].fillna(0).astype(int)
    if "view" in data.columns:
        data["view"] = data["view"].fillna(0).astype(int)
    if "sqft_basement" in data.columns:
        data["sqft_basement"] = data["sqft_basement"].fillna(0).astype(float)

    # Zipcode formatted as clean string
    if "zipcode" in data.columns:
        data["zipcode"] = (
            data["zipcode"]
            .astype(str)
            .str.replace(r"\.0$", "", regex=True)
            .str.strip()
        )

    # Drop duplicates on business key (id is unique per listing)
    if "id" in data.columns:
        data = data.drop_duplicates(subset=["id"], keep="first")

    # Cap unrealistic bedroom counts (data-entry errors like 33 bedrooms)
    data.loc[data["bedrooms"] > 15, "bedrooms"] = 15
    data.loc[data["bedrooms"] < 1, "bedrooms"] = 1

    # yr_renovated: 0 or NaN means never renovated
    data.loc[data["yr_renovated"] == 0, "yr_renovated"] = pd.NA

    # IQR-based capping on price and sqft_living (keep rows, cap extremes)
    for col in ["price", "sqft_living", "sqft_lot"]:
        q1, q3 = data[col].quantile(0.005), data[col].quantile(0.995)
        data[col] = data[col].clip(lower=q1, upper=q3)

    data = data.dropna(subset=[TARGET_COLUMN])
    return data.reset_index(drop=True)


def save_processed_data(df: pd.DataFrame, path: Path | None = None) -> Path:
    path = path or DEFAULT_PROCESSED_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def load_processed_data(path: Path | None = None) -> pd.DataFrame:
    path = path or DEFAULT_PROCESSED_PATH
    if not path.exists():
        raw = load_raw_data()
        cleaned = clean_data(raw)
        save_processed_data(cleaned, path)
    df = pd.read_csv(path)
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
    return df


def get_feature_columns() -> list[str]:
    """Columns used as model inputs after engineering inside the pipeline."""
    return [
        "bedrooms",
        "bathrooms",
        "sqft_living",
        "sqft_lot",
        "floors",
        "waterfront",
        "view",
        "condition",
        "grade",
        "sqft_above",
        "sqft_basement",
        "yr_built",
        "yr_renovated",
        "zipcode",
        "lat",
        "long",
        "sqft_living15",
        "sqft_lot15",
        "sale_year",
    ]


if __name__ == "__main__":
    raw = load_raw_data()
    cleaned = clean_data(raw)
    out = save_processed_data(cleaned)
    print(f"Saved {len(cleaned):,} rows to {out}")
