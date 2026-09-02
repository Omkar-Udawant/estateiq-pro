# 📋 Model Card: EstateIQ Pro (estateiq_model_v1)

## Model Overview
- **Model Name:** EstateIQ Pro v1 (estateiq_model_v1.pkl)
- **Model Architecture:** LightGBM Gradient Boosted Regressor with TransformedTargetRegressor(func=np.log1p, inverse_func=np.expm1)
- **Frameworks:** scikit-learn, LightGBM, Optuna, SHAP
- **Primary Task:** Residential Property Valuation & Explainability
- **Date Trained:** 2026-08-24

## Training Data & Preprocessing
- **Dataset:** King County House Sales (21,420 clean transaction records)
- **Train/Test Split:** 80% Training (17,136 records), 20% Holdout Test (4,284 records), stratified across 10 price deciles.
- **Engineered Features:**
  - house_age: \_year - yr\_built$
  - years_since_renovation: Effective recency of capital renovation
  - ed_bath_ratio: Ratio of bathrooms to bedrooms
  - sqft_living_x_grade: Multiplicative interaction capturing luxury finishes per square foot
  - 
elative_sqft_living: Property size relative to immediate neighborhood average (sqft_living15)
- **Scaling & Encoding:** RobustScaler on numerical features, OneHotEncoder(handle_unknown='ignore') on categorical zipcode.

## Hyperparameter Optimization (Optuna)
- **Method:** Bayesian optimization with Tree-structured Parzen Estimator (TPE)
- **Validation:** 5-Fold Cross Validation optimizing for Mean Absolute Error
- **Optimal Hyperparameters:**
  - 
_estimators: 400
  - 
um_leaves: 80
  - learning_rate: 0.0411
  - eature_fraction: 0.7904
  - agging_fraction: 0.9450
  - agging_freq: 1
  - min_child_samples: 16
  - 
eg_alpha: 0.0128
  - 
eg_lambda: 4.7053

## Evaluation Metrics (Holdout Test Set)
- **R² Score:** 0.9017
- **Mean Absolute Percentage Error (MAPE):** 11.60%
- **Mean Absolute Error (MAE):** ,293
- **Root Mean Squared Error (RMSE):** ,291

## Explainability (SHAP)
- Global and local feature attributions are computed using TreeSHAP (shap.TreeExplainer).
- Top positive value drivers: sqft_living_x_grade, lat, long, grade, sqft_living.
- Local explanations are served in real-time within the Streamlit UI as waterfall plots.

## Intended Use & Limitations
- **Intended Use:** First-pass property valuation, seller listing price guidance, buyer negotiation benchmarking, and automated portfolio screening.
- **Limitations:**
  - Model is trained exclusively on King County, WA transactions (May 2014 - May 2015).
  - Out-of-region zip codes will fall back to general geospatial coordinates.
  - Does not account for macro-economic interest rate shocks without external temporal indexing.
