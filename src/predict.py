"""Inference utilities for EstateIQ Pro."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.data_loader import get_feature_columns

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "estateiq_model_v1.pkl"
DEFAULT_METADATA_PATH = PROJECT_ROOT / "models" / "metadata.json"


class EstateIQPredictor:
    """Load model once and serve predictions with confidence estimates."""

    def __init__(self, model_path: Path | None = None):
        self.model_path = model_path or DEFAULT_MODEL_PATH
        self.pipeline = joblib.load(self.model_path)
        self.feature_columns = get_feature_columns()
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> dict:
        if DEFAULT_METADATA_PATH.exists():
            return json.loads(DEFAULT_METADATA_PATH.read_text())
        return {}

    def _prepare_input(self, features: dict) -> pd.DataFrame:
        row = {col: features.get(col, np.nan) for col in self.feature_columns}
        if "zipcode" in row and row["zipcode"] is not None:
            row["zipcode"] = str(row["zipcode"])
        if "sale_year" not in row or row["sale_year"] is None:
            row["sale_year"] = 2015
        return pd.DataFrame([row])

    def predict(self, features: dict) -> float:
        X = self._prepare_input(features)
        return float(self.pipeline.predict(X)[0])

    def predict_with_confidence(self, features: dict) -> dict:
        """Return prediction with confidence derived from model MAPE."""
        price = self.predict(features)
        test_metrics = self.metadata.get("test_metrics", {})
        mape_pct = test_metrics.get("mape", 8.0)
        mae = test_metrics.get("mae", price * 0.08)

        # Approximate 90% interval using MAE as half-width proxy
        margin = mae * 1.28
        lower = max(0, price - margin)
        upper = price + margin
        interval_width = upper - lower
        confidence = max(0, min(100, 100 - (interval_width / price) * 50))

        sqft = features.get("sqft_living", 1500)
        price_per_sqft = price / sqft if sqft else 0
        market_cat = self.get_market_category(price, sqft, features.get("zipcode", "98103"))

        return {
            "predicted_price": price,
            "confidence_score": round(confidence, 1),
            "price_low": lower,
            "price_high": upper,
            "price_per_sqft": price_per_sqft,
            "market_category": market_cat,
            "mape_reference": mape_pct,
        }

    def get_market_category(self, price: float, sqft: float, zipcode: str | int) -> str:
        """Classify property valuation relative to local zipcode benchmarks."""
        avg_ppsf = self.get_neighborhood_avg_price_per_sqft(str(zipcode))
        actual_ppsf = price / sqft if sqft else 0
        ratio = actual_ppsf / avg_ppsf if avg_ppsf > 0 else 1.0

        if ratio < 0.90:
            return "Under-priced (High Value Opportunity)"
        elif ratio > 1.10:
            return "Premium / Above Market Average"
        return "Fair Market Value"

    def get_neighborhood_avg_price_per_sqft(self, zipcode: str) -> float:
        """Benchmark from processed training dataset."""
        from src.data_loader import load_processed_data, TARGET_COLUMN

        try:
            df = load_processed_data()
            subset = df[df["zipcode"].astype(str) == str(zipcode).replace(".0", "").strip()]
            if subset.empty:
                subset = df
            return float((subset[TARGET_COLUMN] / subset["sqft_living"]).median())
        except Exception:
            return 260.0

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Score multiple properties in batch."""
        res_df = df.copy()
        res_df["predicted_price"] = self.pipeline.predict(df)
        return res_df


def load_predictor(model_path: Path | None = None) -> EstateIQPredictor:
    return EstateIQPredictor(model_path)
