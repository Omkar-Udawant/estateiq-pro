"""Model training, multi-model benchmarking, Bayesian tuning, and artifact persistence."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import joblib
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import seaborn as sns
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.data_loader import TARGET_COLUMN, get_feature_columns, load_processed_data
from src.evaluate import compute_metrics, save_metrics
from src.preprocessing import build_preprocessor, get_transformed_feature_names

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"

optuna.logging.set_verbosity(optuna.logging.WARNING)


def _make_pipeline(model) -> Pipeline:
    """Wrap preprocessor and regressor in a log-target transformed pipeline."""
    return Pipeline(
        [
            ("preprocessor", build_preprocessor()),
            (
                "model",
                TransformedTargetRegressor(
                    regressor=model,
                    func=np.log1p,
                    inverse_func=np.expm1,
                ),
            ),
        ]
    )


def _evaluate_pipeline(pipe: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
    """Evaluate pipeline predictions against ground truth target."""
    preds = pipe.predict(X)
    return compute_metrics(y.values, preds)


def train_baselines(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, dict]:
    """Train candidate baseline models for empirical benchmark comparison."""
    baselines = {
        "Ridge": Ridge(alpha=1.0),
        "RandomForest": RandomForestRegressor(
            n_estimators=100, max_depth=16, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=150, max_depth=6, learning_rate=0.08, random_state=42, n_jobs=-1
        ),
        "LightGBM (default)": lgb.LGBMRegressor(
            n_estimators=150, learning_rate=0.08, random_state=42, n_jobs=-1, verbose=-1
        ),
    }
    results = {}
    for name, estimator in baselines.items():
        print(f"  --> Training {name} baseline...", flush=True)
        pipe = _make_pipeline(estimator)
        pipe.fit(X_train, y_train)
        results[name] = _evaluate_pipeline(pipe, X_test, y_test)
    return results


def tune_lightgbm(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 30,
) -> dict:
    """Fast Bayesian hyperparameter tuning with Optuna using pre-transformed features and 5-Fold CV."""
    preprocessor = build_preprocessor()
    X_train_trans = preprocessor.fit_transform(X_train)
    log_y = np.log1p(y_train.values)
    y_true = y_train.values

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    splits = list(kf.split(X_train_trans))

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 600, step=50),
            "num_leaves": trial.suggest_int("num_leaves", 20, 80),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.12, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.65, 0.95),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.65, 0.95),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 5),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 40),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-2, 5.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 5.0, log=True),
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        mae_scores = []
        for train_idx, val_idx in splits:
            X_tr, y_tr = X_train_trans[train_idx], log_y[train_idx]
            X_va = X_train_trans[val_idx]
            y_va_true = y_true[val_idx]

            model = lgb.LGBMRegressor(**params)
            model.fit(X_tr, y_tr)
            preds = np.expm1(model.predict(X_va))
            mae_scores.append(np.mean(np.abs(y_va_true - preds)))

        return float(np.mean(mae_scores))

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


def generate_evaluation_artifacts(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    test_metrics: dict,
):
    """Generate professional visual artifacts for model interpretability and diagnostics."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", palette="deep")

    # 1. Feature Importances
    try:
        preprocessor = pipeline.named_steps["preprocessor"]
        lgb_model = pipeline.named_steps["model"].regressor_
        importances = lgb_model.feature_importances_
        names = get_transformed_feature_names(preprocessor)

        n_features = min(len(names), len(importances))
        importance_df = pd.DataFrame(
            {"feature": names[:n_features], "importance": importances[:n_features]}
        )
        importance_df = importance_df.sort_values("importance", ascending=False).head(20)

        plt.figure(figsize=(10, 7))
        bars = plt.barh(importance_df["feature"], importance_df["importance"], color="#2b6cb0")
        plt.gca().invert_yaxis()
        plt.xlabel("Split Feature Importance (LightGBM)", fontsize=11, fontweight="bold")
        plt.title("EstateIQ Pro — Top 20 Feature Importances", fontsize=13, fontweight="bold", pad=12)
        plt.tight_layout()
        plt.savefig(ARTIFACTS_DIR / "feature_importance.png", dpi=150)
        plt.close()
        print("  [Artifact] Saved artifacts/feature_importance.png")
    except Exception as exc:
        print(f"  [Warning] Feature importance plot failed: {exc}")

    # 2. Actual vs. Predicted Scatter
    try:
        preds = pipeline.predict(X_test)
        plt.figure(figsize=(8, 8))
        plt.scatter(y_test / 1000, preds / 1000, alpha=0.35, color="#2b6cb0", edgecolors="none", s=25)
        min_val = min(y_test.min(), preds.min()) / 1000
        max_val = max(y_test.max(), preds.max()) / 1000
        plt.plot([min_val, max_val], [min_val, max_val], color="#e53e3e", linestyle="--", lw=2, label="Perfect Fit")
        plt.xlabel("Actual Price ($k)", fontsize=11, fontweight="bold")
        plt.ylabel("Predicted Price ($k)", fontsize=11, fontweight="bold")
        plt.title(
            f"EstateIQ Pro — Actual vs Predicted (R² = {test_metrics['r2']:.3f}, MAPE = {test_metrics['mape']:.1f}%)",
            fontsize=12,
            fontweight="bold",
            pad=12,
        )
        plt.legend(frameon=True)
        plt.tight_layout()
        plt.savefig(ARTIFACTS_DIR / "actual_vs_predicted.png", dpi=150)
        plt.close()
        print("  [Artifact] Saved artifacts/actual_vs_predicted.png")
    except Exception as exc:
        print(f"  [Warning] Actual vs predicted plot failed: {exc}")

    # 3. SHAP Summary Plot
    try:
        import shap

        preprocessor = pipeline.named_steps["preprocessor"]
        sample_X = X_test.sample(min(400, len(X_test)), random_state=42)
        sample_transformed = preprocessor.transform(sample_X)
        lgb_model = pipeline.named_steps["model"].regressor_

        explainer = shap.TreeExplainer(lgb_model)
        shap_values = explainer.shap_values(sample_transformed)
        feature_names = get_transformed_feature_names(preprocessor)

        plt.figure(figsize=(10, 7))
        shap.summary_plot(
            shap_values,
            sample_transformed,
            feature_names=feature_names[: sample_transformed.shape[1]],
            max_display=15,
            show=False,
        )
        plt.title("EstateIQ Pro — Global SHAP Feature Impact", fontsize=13, fontweight="bold", pad=15)
        plt.tight_layout()
        plt.savefig(ARTIFACTS_DIR / "shap_summary.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("  [Artifact] Saved artifacts/shap_summary.png")
    except Exception as exc:
        print(f"  [Warning] SHAP summary plot failed: {exc}")


def train_production_model(
    n_trials: int = 30,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Path:
    """Full training workflow: baselines → Optuna → save model & metadata."""
    print("=" * 65)
    print("EstateIQ Pro -- Production Training Pipeline")
    print("=" * 65)

    df = load_processed_data()
    features = get_feature_columns()
    X = df[features]
    y = df[TARGET_COLUMN]

    # Price-bucket stratification via quantiles for balanced holdout
    price_bins = pd.qcut(y, q=10, duplicates="drop")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=price_bins
    )
    print(f"Dataset split: {len(X_train):,} training records, {len(X_test):,} holdout test records\n")

    print("Step 1/3: Training Baseline Models for Benchmark Comparison...")
    baseline_metrics = train_baselines(X_train, y_train, X_test, y_test)
    print("\n--- Baseline Benchmarks ---")
    for name, metrics in baseline_metrics.items():
        print(f"  - {name:<20}: R2 = {metrics['r2']:.4f} | MAPE = {metrics['mape']:.2f}% | MAE = ${metrics['mae']:,.0f}")

    print(f"\nStep 2/3: Bayesian Hyperparameter Optimization with Optuna ({n_trials} trials, 5-Fold CV)...")
    best_params = tune_lightgbm(X_train, y_train, n_trials=n_trials)
    print(f"  --> Optimal Parameters Identified: {best_params}")

    print("\nStep 3/3: Fitting Production Model with Optimal Parameters...")
    final_model = lgb.LGBMRegressor(**best_params, random_state=42, n_jobs=-1, verbose=-1)
    pipeline = _make_pipeline(final_model)
    pipeline.fit(X_train, y_train)

    test_metrics = _evaluate_pipeline(pipeline, X_test, y_test)
    print("\n" + "=" * 45)
    print("CHAMPION MODEL METRICS (HOLDOUT TEST):")
    print(f"   MAE:  ${test_metrics['mae']:,.0f}")
    print(f"   RMSE: ${test_metrics['rmse']:,.0f}")
    print(f"   MAPE: {test_metrics['mape']:.2f}%")
    print(f"   R2:   {test_metrics['r2']:.4f}")
    print("=" * 45 + "\n")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODELS_DIR / "estateiq_model_v1.pkl"
    joblib.dump(pipeline, model_path)

    metadata = {
        "model_name": "estateiq_model_v1",
        "algorithm": "LightGBM Regressor (Optuna-tuned)",
        "target_transform": "np.log1p",
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "best_params": best_params,
        "baseline_metrics": baseline_metrics,
        "test_metrics": test_metrics,
        "feature_columns": features,
    }
    metadata_path = MODELS_DIR / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2))
    save_metrics(test_metrics)

    print("Generating Visual Diagnostic & Explainability Artifacts...")
    generate_evaluation_artifacts(pipeline, X_train, X_test, y_test, test_metrics)

    print(f"\n[OK] Production model serialized to: {model_path}")
    print(f"[OK] Model metadata saved to: {metadata_path}")
    return model_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train EstateIQ Pro model")
    parser.add_argument("--trials", type=int, default=30, help="Optuna trials")
    args = parser.parse_args()
    train_production_model(n_trials=args.trials)

