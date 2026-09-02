"""Sklearn preprocessing pipeline with feature engineering."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from src.data_loader import CATEGORICAL_COLUMNS, get_feature_columns


class ColumnSelector(BaseEstimator, TransformerMixin):
    """Select and order columns from raw inputs, adding NaN for missing."""

    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        for col in self.columns:
            if col not in df.columns:
                df[col] = np.nan
        return df[self.columns]


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """Business-driven feature creation before encoding/scaling."""

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        sale_year = df.get("sale_year", 2015)
        if not isinstance(sale_year, pd.Series):
            sale_year = pd.Series([sale_year] * len(df), index=df.index)
        else:
            sale_year = sale_year.fillna(2015)

        yr_built = pd.to_numeric(df["yr_built"], errors="coerce").fillna(1975)
        df["house_age"] = np.clip(sale_year - yr_built, 0, 150)

        reno = pd.to_numeric(df["yr_renovated"], errors="coerce").fillna(0)
        df["is_renovated"] = (reno > 0).astype(int)
        df["years_since_renovation"] = np.where(
            reno > 0,
            np.clip(sale_year - reno, 0, 150),
            df["house_age"],
        )

        sqft_basement = pd.to_numeric(df["sqft_basement"], errors="coerce").fillna(0)
        df["has_basement"] = (sqft_basement > 0).astype(int)

        bedrooms = pd.to_numeric(df["bedrooms"], errors="coerce").fillna(3).clip(lower=1)
        bathrooms = pd.to_numeric(df["bathrooms"], errors="coerce").fillna(2).clip(lower=0.5)
        sqft_living = pd.to_numeric(df["sqft_living"], errors="coerce").fillna(1800).clip(lower=100)
        sqft_lot = pd.to_numeric(df["sqft_lot"], errors="coerce").fillna(5000).clip(lower=100)
        sqft_above = pd.to_numeric(df["sqft_above"], errors="coerce").fillna(sqft_living)
        grade = pd.to_numeric(df["grade"], errors="coerce").fillna(7)
        condition = pd.to_numeric(df["condition"], errors="coerce").fillna(3)
        sqft_living15 = pd.to_numeric(df["sqft_living15"], errors="coerce").fillna(sqft_living).clip(lower=100)
        sqft_lot15 = pd.to_numeric(df["sqft_lot15"], errors="coerce").fillna(sqft_lot).clip(lower=100)

        df["bed_bath_ratio"] = bathrooms / bedrooms
        df["sqft_ratio_living_lot"] = sqft_living / sqft_lot
        df["sqft_per_bedroom"] = sqft_living / bedrooms
        df["sqft_living_x_grade"] = sqft_living * grade
        df["sqft_living_x_condition"] = sqft_living * condition
        df["sqft_above_ratio"] = sqft_above / sqft_living
        df["relative_sqft_living"] = sqft_living / sqft_living15
        df["relative_sqft_lot"] = sqft_lot / sqft_lot15

        return df


ENGINEERED_NUMERIC = [
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
    "lat",
    "long",
    "sqft_living15",
    "sqft_lot15",
    "house_age",
    "is_renovated",
    "years_since_renovation",
    "has_basement",
    "bed_bath_ratio",
    "sqft_ratio_living_lot",
    "sqft_per_bedroom",
    "sqft_living_x_grade",
    "sqft_living_x_condition",
    "sqft_above_ratio",
    "relative_sqft_living",
    "relative_sqft_lot",
]


def build_preprocessor() -> Pipeline:
    """Full preprocessing pipeline: select inputs -> engineer -> transform."""
    input_columns = get_feature_columns()

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    column_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, ENGINEERED_NUMERIC),
            ("cat", categorical_pipeline, CATEGORICAL_COLUMNS),
        ],
        remainder="drop",
    )

    return Pipeline(
        [
            ("select", ColumnSelector(input_columns)),
            ("engineer", FeatureEngineer()),
            ("transform", column_transformer),
        ]
    )


def get_transformed_feature_names(preprocessor: Pipeline) -> list[str]:
    """Human-readable feature names after preprocessing."""
    ct: ColumnTransformer = preprocessor.named_steps["transform"]
    cat_encoder = ct.named_transformers_["cat"].named_steps["encoder"]
    cat_names = list(cat_encoder.get_feature_names_out(CATEGORICAL_COLUMNS))
    return ENGINEERED_NUMERIC + cat_names
