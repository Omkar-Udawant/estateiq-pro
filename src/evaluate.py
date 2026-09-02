"""Evaluation metrics for EstateIQ Pro."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error (%), avoiding division by zero."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true > 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute business-relevant regression metrics."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": rmse,
        "mape": mape(y_true, y_pred),
        "r2": float(r2_score(y_true, y_pred)),
    }


def save_metrics(metrics: dict, path: Path | None = None) -> Path:
    path = path or ARTIFACTS_DIR / "metrics.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(metrics, indent=2))
    return path


def load_metrics(path: Path | None = None) -> dict:
    path = path or ARTIFACTS_DIR / "metrics.json"
    return json.loads(path.read_text())


def format_metrics_report(metrics: dict) -> str:
    return (
        f"MAE:  ${metrics['mae']:,.0f}\n"
        f"RMSE: ${metrics['rmse']:,.0f}\n"
        f"MAPE: {metrics['mape']:.2f}%\n"
        f"R²:   {metrics['r2']:.4f}"
    )
