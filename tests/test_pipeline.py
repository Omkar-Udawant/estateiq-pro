"""Unit and integration tests for EstateIQ Pro pipeline."""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import clean_data, get_feature_columns
from src.evaluate import compute_metrics, mape
from src.predict import load_predictor
from src.preprocessing import FeatureEngineer, build_preprocessor


@pytest.fixture
def sample_raw_df():
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "date": ["20140502", "20140603", "20140704"],
            "price": [500000, 600000, 450000],
            "bedrooms": [3, 33, 2],
            "bathrooms": [2.0, 2.5, 1.0],
            "sqft_living": [1800, 2200, 1200],
            "sqft_lot": [5000, 6000, 4000],
            "floors": [2.0, 2.0, 1.0],
            "waterfront": [0, 0, 1],
            "view": [0, 2, 0],
            "condition": [3, 4, 3],
            "grade": [7, 8, 6],
            "sqft_above": [1800, 2200, 1200],
            "sqft_basement": ["0", "0", "?"],
            "yr_built": [1990, 2000, 1980],
            "yr_renovated": [0, 2010, 0],
            "zipcode": [98103, 98115, "98001.0"],
            "lat": [47.66, 47.68, 47.30],
            "long": [-122.34, -122.30, -122.20],
            "sqft_living15": [1700, 2100, 1300],
            "sqft_lot15": [4800, 5900, 4100],
        }
    )


def test_clean_data_caps_bedrooms(sample_raw_df):
    cleaned = clean_data(sample_raw_df)
    assert cleaned["bedrooms"].max() <= 15
    assert cleaned["bedrooms"].min() >= 1


def test_clean_data_handles_question_marks_and_zipcode_strings(sample_raw_df):
    cleaned = clean_data(sample_raw_df)
    assert cleaned["sqft_basement"].dtype in [np.float64, np.int64, float]
    assert cleaned["zipcode"].iloc[2] == "98001"


def test_feature_engineer_adds_columns():
    df = pd.DataFrame(
        {
            "yr_built": [1990],
            "yr_renovated": [pd.NA],
            "bedrooms": [3],
            "bathrooms": [2.0],
            "sqft_living": [2000],
            "sqft_lot": [5000],
            "sqft_above": [2000],
            "sqft_basement": [0],
            "grade": [8],
            "condition": [3],
            "sqft_living15": [1900],
            "sqft_lot15": [4800],
            "sale_year": [2015],
        }
    )
    out = FeatureEngineer().transform(df)
    assert "house_age" in out.columns
    assert "bed_bath_ratio" in out.columns
    assert "sqft_living_x_grade" in out.columns
    assert out["house_age"].iloc[0] == 25
    assert out["sqft_living_x_grade"].iloc[0] == 16000


def test_preprocessor_fit_transform(sample_raw_df):
    cleaned = clean_data(sample_raw_df)
    features = get_feature_columns()
    X = cleaned[features]
    y = cleaned["price"]
    pipe = build_preprocessor()
    transformed = pipe.fit_transform(X, y)
    assert transformed.shape[0] == len(X)
    assert transformed.shape[1] > 15


def test_compute_metrics_perfect_prediction():
    y = np.array([100000, 200000, 300000])
    metrics = compute_metrics(y, y)
    assert metrics["mae"] == 0.0
    assert metrics["r2"] == 1.0
    assert metrics["mape"] == 0.0


def test_mape_handles_zeros():
    y_true = np.array([0, 100000])
    y_pred = np.array([50000, 90000])
    result = mape(y_true, y_pred)
    assert result == pytest.approx(10.0)


def test_predictor_single_and_batch_inference():
    predictor = load_predictor()
    single_prop = {
        "sqft_living": 2200,
        "bedrooms": 3,
        "bathrooms": 2.5,
        "yr_built": 1995,
        "grade": 8,
        "zipcode": "98103",
    }
    res = predictor.predict_with_confidence(single_prop)
    assert res["predicted_price"] > 100000
    assert 0 <= res["confidence_score"] <= 100
    assert res["price_low"] < res["predicted_price"] < res["price_high"]
    assert "market_category" in res

    batch_df = pd.DataFrame([single_prop, single_prop])
    batch_res = predictor.predict_batch(batch_df)
    assert len(batch_res) == 2
    assert "predicted_price" in batch_res.columns

