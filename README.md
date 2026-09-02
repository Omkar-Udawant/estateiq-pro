# 🏠 EstateIQ Pro — AI Real Estate Valuation Platform

> Enterprise-grade Machine Learning Valuation & Explainability System for King County, WA

[![Live Demo](https://img.shields.io/badge/Live%20Demo-estateiq--pro.streamlit.app-FF4B4B?style=for-the-badge&logo=streamlit)](https://estateiq-pro.streamlit.app)

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://estateiq-pro.streamlit.app)
[![LightGBM](https://img.shields.io/badge/LightGBM-Optuna%20Tuned-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![Tests](https://img.shields.io/badge/Tests-100%25%20Passing-success.svg)](https://pytest.org/)

🔗 **Live Interactive Web App:** [https://estateiq-pro.streamlit.app](https://estateiq-pro.streamlit.app)

---

## 📌 Executive Summary

Property mispricing costs home buyers, sellers, and real estate brokerages tens of thousands of dollars per transaction. Overpriced listings linger on market 30–90+ days longer, while underpriced properties forfeit equity. 

**EstateIQ Pro** is a production-grade ML application that replaces slow, subjective appraisals with instantaneous, explainable property valuations powered by **LightGBM, XGBoost, and SHAP**.

```
User Input / Web UI (Streamlit)
        │
        ▼
Preprocessing Pipeline (src/preprocessing.py)
  ├─ Business Feature Engineering (house_age, renovation recency, bed/bath ratio, sqft x grade)
  ├─ RobustScaler (outlier resilience)
  └─ OneHotEncoder (zipcode categorical encoding)
        │
        ▼
Optuna-Tuned LightGBM Regressor (TransformedTarget np.log1p)
        │
        ▼
Inference & Explainability Engine (src/predict.py)
  ├─ Point Price Valuation ($)
  ├─ 90% Confidence Interval Bounds ($)
  ├─ Neighborhood Benchmark ($/sqft vs Zipcode Median)
  └─ SHAP Local Feature Attribution Waterfall
```

---

## 🏆 Model Benchmarking & Performance

Every candidate algorithm was benchmarked on a stratified holdout test split (4,284 records, 20% holdout):

| Algorithm | R² Score | MAPE (%) | MAE ($) | RMSE ($) |
|---|---|---|---|---|
| Ridge Regression (Baseline) | 0.8632 | 13.45% | $72,384 | $124,190 |
| Random Forest Regressor | 0.8758 | 12.42% | $67,376 | $118,333 |
| XGBoost Regressor | 0.8925 | 12.03% | $64,046 | $110,087 |
| LightGBM (Default) | 0.8935 | 11.94% | $63,838 | $109,590 |
| 🏆 **LightGBM (Optuna Tuned — Production)** | **0.9017** | **11.60%** | **$61,293** | **$105,291** |

---

## 📂 Project Architecture

```
estateiq-pro/
├── data/
│   ├── raw/                  # Original King County dataset (kc_house_data.csv)
│   └── processed/            # Cleaned data (kc_house_clean.csv)
├── notebooks/
│   └── 01_eda_and_insights.ipynb  # End-to-end exploratory analysis & market patterns
├── src/
│   ├── data_loader.py        # Automated ingestion & cleaning
│   ├── preprocessing.py      # scikit-learn Pipeline & FeatureEngineer
│   ├── train.py              # Multi-model benchmarking & Optuna Bayesian tuning
│   ├── evaluate.py           # Regression metrics (MAE, RMSE, MAPE, R²)
│   └── predict.py            # Production inference engine & confidence bounds
├── models/
│   ├── estateiq_model_v1.pkl # Serialized production pipeline
│   └── metadata.json         # Benchmark metrics, hyperparameters, and feature metadata
├── artifacts/
│   ├── metrics.json          # Final test set metrics
│   ├── feature_importance.png# Top 20 feature importances
│   ├── actual_vs_predicted.png# Actual vs predicted scatter plot
│   └── shap_summary.png      # Global SHAP attribution summary
├── app/
│   └── streamlit_app.py      # Multi-tab interactive SaaS web application
├── tests/
│   ├── conftest.py           # Pytest path resolution
│   └── test_pipeline.py      # Unit and integration test suite
├── MODEL_CARD.md             # Model specification, intended use, and limitations
├── requirements.txt          # Pinned production dependencies
└── streamlit_app.py          # Streamlit Community Cloud deployment entry point
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup
```bash
git clone https://github.com/yourusername/estateiq-pro.git
cd estateiq-pro
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

### 2. Ingest & Clean Dataset
```bash
python -m src.data_loader
```

### 3. Train & Tune Production Model
```bash
python -m src.train --trials 30
```

### 4. Run Automated Test Suite
```bash
python -m pytest tests/ -v
```

### 5. Launch Interactive Dashboard
```bash
streamlit run streamlit_app.py
```

---

## 💡 Key Engineering Highlights

1. **Target Transformed Loss (`np.log1p`):** Solves heavy right-skewed property price distributions, aligning model optimization with percentage accuracy across median-priced homes.
2. **Leakage-Free `scikit-learn` Pipeline:** Encapsulates feature creation, imputation, scaling, and one-hot encoding into a unified pipeline fitted strictly on training splits.
3. **Optuna Bayesian Optimization:** Systematically searched 30 parameter combinations across 5-fold CV to boost R² from 0.86 (Ridge baseline) to 0.9017.
4. **SHAP Per-Prediction Attribution:** Live waterfall visualizations in Streamlit show buyers and sellers the exact dollar impact of square footage, condition, grade, and location.
5. **Interactive Geospatial Dashboard:** Multi-tab layout featuring quick presets, King County map comparables, and exportable HTML valuation reports.

---

## 📜 License

Distributed under the MIT License. See `LICENSE` for more information.

