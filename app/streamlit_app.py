"""EstateIQ Pro — Industry-Grade AI House Price Valuation Platform."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.data_loader import load_processed_data
from src.predict import load_predictor

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "estateiq_model_v1.pkl"
METADATA_PATH = PROJECT_ROOT / "models" / "metadata.json"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"


# Zipcode coordinates cache for auto-populating lat/long defaults
ZIP_COORDS = {
    "98001": (47.30, -122.26), "98002": (47.31, -122.21), "98003": (47.31, -122.31),
    "98004": (47.62, -122.20), "98005": (47.61, -122.17), "98006": (47.56, -122.16),
    "98007": (47.61, -122.14), "98008": (47.60, -122.12), "98010": (47.32, -122.01),
    "98011": (47.76, -122.21), "98014": (47.65, -121.90), "98019": (47.74, -121.96),
    "98022": (47.20, -122.00), "98023": (47.31, -122.36), "98024": (47.57, -121.89),
    "98027": (47.53, -122.03), "98028": (47.76, -122.26), "98029": (47.55, -122.00),
    "98030": (47.37, -122.18), "98031": (47.39, -122.20), "98032": (47.39, -122.26),
    "98033": (47.68, -122.19), "98034": (47.71, -122.21), "98038": (47.37, -122.03),
    "98039": (47.63, -122.24), "98040": (47.57, -122.23), "98042": (47.37, -122.12),
    "98045": (47.49, -121.78), "98052": (47.67, -122.12), "98053": (47.66, -122.02),
    "98055": (47.46, -122.21), "98056": (47.50, -122.18), "98058": (47.44, -122.16),
    "98059": (47.49, -122.13), "98065": (47.53, -121.87), "98070": (47.41, -122.46),
    "98072": (47.75, -122.13), "98074": (47.62, -122.04), "98075": (47.59, -122.04),
    "98077": (47.74, -122.06), "98092": (47.31, -122.20), "98102": (47.63, -122.32),
    "98103": (47.67, -122.34), "98105": (47.66, -122.28), "98106": (47.53, -122.36),
    "98107": (47.67, -122.37), "98108": (47.55, -122.32), "98109": (47.63, -122.35),
    "98112": (47.63, -122.29), "98115": (47.68, -122.30), "98116": (47.58, -122.39),
    "98117": (47.69, -122.37), "98118": (47.54, -122.27), "98119": (47.64, -122.36),
    "98122": (47.61, -122.30), "98125": (47.72, -122.30), "98126": (47.55, -122.37),
    "98133": (47.74, -122.34), "98136": (47.54, -122.39), "98144": (47.59, -122.30),
    "98146": (47.50, -122.36), "98148": (47.44, -122.33), "98155": (47.76, -122.29),
    "98166": (47.44, -122.35), "98168": (47.49, -122.30), "98177": (47.74, -122.37),
    "98178": (47.50, -122.24), "98188": (47.45, -122.28), "98198": (47.39, -122.32),
    "98199": (47.65, -122.40),
}


@st.cache_resource
def get_predictor():
    if not MODEL_PATH.exists():
        st.error("Model artifact not found. Please run `python -m src.train` first.")
        st.stop()
    return load_predictor(MODEL_PATH)


@st.cache_data
def get_dataset_sample():
    df = load_processed_data()
    return df.sample(min(1500, len(df)), random_state=42)


@st.cache_data
def get_metadata() -> dict:
    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text())
    return {}


def render_shap_waterfall(features: dict, predictor):
    """Render SHAP waterfall for single-property attribution."""
    try:
        import shap

        pipeline = predictor.pipeline
        preprocessor = pipeline.named_steps["preprocessor"]
        lgb_model = pipeline.named_steps["model"].regressor_

        X_df = predictor._prepare_input(features)
        X_trans = preprocessor.transform(X_df)

        from src.preprocessing import get_transformed_feature_names

        feature_names = get_transformed_feature_names(preprocessor)
        feature_names_trimmed = feature_names[: X_trans.shape[1]]

        explainer = shap.TreeExplainer(lgb_model)
        shap_vals = explainer.shap_values(X_trans)

        fig, ax = plt.subplots(figsize=(9, 6))
        explanation = shap.Explanation(
            values=shap_vals[0],
            base_values=explainer.expected_value,
            data=X_trans[0],
            feature_names=feature_names_trimmed,
        )
        shap.plots.waterfall(explanation, max_display=12, show=False)
        plt.title("Per-Property Feature Impact (SHAP Values in Log-Space)", fontsize=11, fontweight="bold", pad=12)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    except Exception as exc:
        st.info(f"SHAP waterfall rendering note: {exc}")


def build_report_html(result: dict, features: dict) -> str:
    """Generate a clean, self-contained HTML valuation report."""
    rows = "".join(
        f"<tr><td style='padding:8px;border-bottom:1px solid #e2e8f0;font-weight:600;color:#4a5568;'>{k}</td>"
        f"<td style='padding:8px;border-bottom:1px solid #e2e8f0;color:#2d3748;'>{v}</td></tr>"
        for k, v in features.items()
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>EstateIQ Pro — Property Valuation Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background:#f7fafc; color:#2d3748; padding:30px; }}
.card {{ background:#ffffff; max-width:760px; margin:0 auto; padding:32px; border-radius:12px; box-shadow:0 4px 6px rgba(0,0,0,0.07); border:1px solid #e2e8f0; }}
.header {{ border-bottom:2px solid #3182ce; padding-bottom:16px; margin-bottom:24px; }}
.price {{ font-size:36px; font-weight:800; color:#2b6cb0; margin:8px 0; }}
.badge {{ display:inline-block; padding:4px 12px; border-radius:16px; font-size:13px; font-weight:700; background:#ebf8ff; color:#2b6cb0; }}
table {{ width:100%; border-collapse:collapse; margin-top:20px; }}
.footer {{ margin-top:32px; font-size:12px; color:#a0aec0; text-align:center; border-top:1px solid #e2e8f0; padding-top:16px; }}
</style>
</head>
<body>
<div class="card">
  <div class="header">
    <h1 style="margin:0;color:#1a365d;">🏠 EstateIQ Pro Valuation Report</h1>
    <p style="margin:4px 0 0 0;color:#718096;font-size:14px;">Automated Proptech Machine Learning Valuation</p>
  </div>

  <div class="price">${result['predicted_price']:,.0f}</div>
  <div class="badge">Confidence Score: {result['confidence_score']}%</div>
  <div class="badge" style="background:#fefcbf;color:#b7791f;margin-left:8px;">{result['market_category']}</div>

  <p style="margin:16px 0;font-size:15px;color:#4a5568;">
    <strong>Estimated 90% Valuation Range:</strong> ${result['price_low']:,.0f} &ndash; ${result['price_high']:,.0f}<br/>
    <strong>Valuation Benchmark:</strong> ${result['price_per_sqft']:,.0f} / sq.ft.
  </p>

  <h3 style="color:#2d3748;margin-top:28px;border-bottom:1px solid #e2e8f0;padding-bottom:8px;">Property Specification</h3>
  <table>
    {rows}
  </table>

  <div class="footer">
    EstateIQ Pro &bull; Gradient Boosted Tree Architecture with Bayesian Hyperparameter Optimization &bull; King County, WA
  </div>
</div>
</body>
</html>"""


def main():
    st.set_page_config(
        page_title="EstateIQ Pro — AI Real Estate Valuation",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Top Navbar & Branding
    st.markdown(
        """
        <div style="padding: 10px 0 20px 0;">
            <h1 style="margin:0; font-size: 2.3rem; color: #1e3a8a;">🏠 EstateIQ Pro</h1>
            <p style="margin: 2px 0 0 0; color: #4b5563; font-size: 1.05rem;">
                Enterprise-grade House Price Prediction & Explainability Platform &bull; King County, WA
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    predictor = get_predictor()
    meta = get_metadata()

    # Sidebar: Presets & Controls
    with st.sidebar:
        st.markdown("### ⚙️ Property Configuration")

        preset = st.selectbox(
            "Quick Load Presets",
            [
                "Custom Configuration",
                "Seattle Modern Townhouse (Ballard 98107)",
                "Bellevue Luxury Estate (98004)",
                "Redmond Family Home (98052)",
                "Lake Washington Waterfront Villa (98039)",
            ],
        )

        # Preset values defaults
        defaults = {
            "sqft_living": 2100, "bedrooms": 3, "bathrooms": 2.5, "floors": 2.0,
            "waterfront": 0, "view": 0, "condition": 3, "grade": 8, "sqft_lot": 5000,
            "sqft_above": 1800, "sqft_basement": 300, "yr_built": 1998, "yr_renovated": 0,
            "zipcode": "98103",
        }

        if preset == "Seattle Modern Townhouse (Ballard 98107)":
            defaults.update({"sqft_living": 1650, "bedrooms": 3, "bathrooms": 2.5, "floors": 3.0, "grade": 8, "sqft_lot": 1800, "sqft_above": 1650, "sqft_basement": 0, "yr_built": 2012, "zipcode": "98107"})
        elif preset == "Bellevue Luxury Estate (98004)":
            defaults.update({"sqft_living": 4200, "bedrooms": 4, "bathrooms": 4.5, "floors": 2.0, "grade": 11, "condition": 4, "sqft_lot": 12000, "sqft_above": 3500, "sqft_basement": 700, "yr_built": 2008, "zipcode": "98004"})
        elif preset == "Redmond Family Home (98052)":
            defaults.update({"sqft_living": 2600, "bedrooms": 4, "bathrooms": 2.5, "floors": 2.0, "grade": 8, "condition": 4, "sqft_lot": 8500, "sqft_above": 2600, "sqft_basement": 0, "yr_built": 1994, "zipcode": "98052"})
        elif preset == "Lake Washington Waterfront Villa (98039)":
            defaults.update({"sqft_living": 5400, "bedrooms": 5, "bathrooms": 5.0, "floors": 2.0, "waterfront": 1, "view": 4, "grade": 12, "condition": 5, "sqft_lot": 22000, "sqft_above": 4200, "sqft_basement": 1200, "yr_built": 2010, "zipcode": "98039"})

        zipcode_list = sorted(list(ZIP_COORDS.keys()))
        zip_index = zipcode_list.index(defaults["zipcode"]) if defaults["zipcode"] in zipcode_list else 0

        zipcode = st.selectbox("Zip Code", zipcode_list, index=zip_index)
        default_lat, default_long = ZIP_COORDS.get(zipcode, (47.56, -122.20))

        sqft_living = st.number_input("Living Area (sq.ft.)", 300, 15000, defaults["sqft_living"], step=50)
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            bedrooms = st.slider("Bedrooms", 1, 9, defaults["bedrooms"])
        with col_b2:
            bathrooms = st.slider("Bathrooms", 1.0, 7.0, float(defaults["bathrooms"]), 0.25)

        grade = st.slider("Construction Grade (1-13)", 1, 13, defaults["grade"], help="King County building construction & design grade")
        condition = st.slider("Condition (1-5)", 1, 5, defaults["condition"])
        floors = st.selectbox("Floors", [1.0, 1.5, 2.0, 2.5, 3.0], index=[1.0, 1.5, 2.0, 2.5, 3.0].index(defaults["floors"]))

        with st.expander("Additional Property Features", expanded=False):
            waterfront = st.selectbox("Waterfront View", [0, 1], index=defaults["waterfront"], format_func=lambda x: "Yes (Waterfront)" if x == 1 else "No")
            view = st.slider("View Quality Score", 0, 4, defaults["view"])
            sqft_lot = st.number_input("Lot Size (sq.ft.)", 500, 200000, defaults["sqft_lot"], step=250)
            sqft_above = st.number_input("Above-Ground Area (sq.ft.)", 300, 15000, defaults["sqft_above"], step=50)
            sqft_basement = st.number_input("Basement Area (sq.ft.)", 0, 8000, defaults["sqft_basement"], step=50)
            yr_built = st.number_input("Year Built", 1900, 2025, defaults["yr_built"])
            yr_renovated = st.number_input("Year Renovated (0 = never)", 0, 2025, defaults["yr_renovated"])
            lat = st.number_input("Latitude", 47.0, 48.0, default_lat, format="%.4f")
            long = st.number_input("Longitude", -122.6, -121.0, default_long, format="%.4f")

        predict_clicked = st.button("🚀 Calculate Property Valuation", type="primary", use_container_width=True)

        if meta.get("test_metrics"):
            tm = meta["test_metrics"]
            st.markdown("---")
            st.markdown("##### 📊 Production Model Stats")
            c1, c2 = st.columns(2)
            c1.metric("Holdout R²", f"{tm['r2']:.3f}")
            c2.metric("Holdout MAPE", f"{tm['mape']:.1f}%")

    # Feature input dictionary
    features = {
        "sqft_living": sqft_living,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "floors": floors,
        "waterfront": waterfront,
        "view": view,
        "condition": condition,
        "grade": grade,
        "sqft_lot": sqft_lot,
        "sqft_above": sqft_above,
        "sqft_basement": sqft_basement,
        "yr_built": yr_built,
        "yr_renovated": yr_renovated if yr_renovated > 0 else None,
        "zipcode": zipcode,
        "lat": lat,
        "long": long,
        "sqft_living15": sqft_living,
        "sqft_lot15": sqft_lot,
        "sale_year": 2015,
    }

    result = predictor.predict_with_confidence(features)
    avg_ppsf = predictor.get_neighborhood_avg_price_per_sqft(zipcode)

    # Main application tabs
    tab_valuation, tab_geospatial, tab_explain, tab_model, tab_report = st.tabs(
        [
            "💰 Instant Valuation",
            "🗺️ Geospatial & Comps",
            "🔍 AI Explainability (SHAP)",
            "📈 Model Benchmarks & Metrics",
            "📄 Valuation Report Export",
        ]
    )

    # ----------------------------------------------------
    # TAB 1: Instant Valuation
    # ----------------------------------------------------
    with tab_valuation:
        st.subheader("Property Valuation Summary")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Estimated Price", f"${result['predicted_price']:,.0f}")
        m2.metric("Confidence Score", f"{result['confidence_score']}%")
        m3.metric("Valuation / Sq.Ft.", f"${result['price_per_sqft']:,.0f}")
        m4.metric("Market Status", result["market_category"].split("(")[0].strip())

        st.info(
            f"🎯 **Estimated 90% Confidence Interval:** **${result['price_low']:,.0f}** &nbsp;to&nbsp; **${result['price_high']:,.0f}** "
            f"(Derived from empirical holdout residual distribution with ±{result['mape_reference']:.1f}% MAPE margin)"
        )

        col_ch1, col_ch2 = st.columns(2)

        with col_ch1:
            st.markdown("##### 📊 Price / Sq.Ft. vs Zip Code Benchmark")
            fig_bar = go.Figure(
                data=[
                    go.Bar(
                        x=["This Property", f"Zip {zipcode} Median"],
                        y=[result["price_per_sqft"], avg_ppsf],
                        marker_color=["#1d4ed8", "#94a3b8"],
                        text=[f"${result['price_per_sqft']:,.0f}", f"${avg_ppsf:,.0f}"],
                        textposition="auto",
                    )
                ]
            )
            fig_bar.update_layout(
                yaxis_title="USD ($) per Square Foot",
                height=320,
                margin=dict(l=20, r=20, t=30, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        with col_ch2:
            st.markdown("##### 🧭 Property Specification Breakdown")
            spec_df = pd.DataFrame(
                {
                    "Metric": [
                        "Living Area", "Bed / Bath Ratio", "Construction Grade",
                        "Condition Score", "Effective House Age", "Renovation Status",
                    ],
                    "Value": [
                        f"{sqft_living:,} sqft",
                        f"{bathrooms/bedrooms:.2f}",
                        f"{grade} / 13",
                        f"{condition} / 5",
                        f"{2015 - yr_built} years",
                        f"Renovated {yr_renovated}" if yr_renovated > 0 else "Original",
                    ],
                }
            )
            st.dataframe(spec_df, hide_index=True, use_container_width=True)

    # ----------------------------------------------------
    # TAB 2: Geospatial & Comps
    # ----------------------------------------------------
    with tab_geospatial:
        st.subheader("Geospatial Valuation Context in King County")
        st.caption("Locating your subject property against comparable historical listings in King County, WA.")

        sample_df = get_dataset_sample()
        map_df = sample_df.copy()
        map_df["PriceTier"] = pd.qcut(map_df["price"], q=4, labels=["Budget", "Mid-Range", "Upscale", "Luxury"])

        fig_map = px.scatter_mapbox(
            map_df,
            lat="lat",
            lon="long",
            color="PriceTier",
            size="sqft_living",
            hover_name="zipcode",
            hover_data={"price": ":$,.0f", "sqft_living": True, "lat": False, "long": False},
            color_discrete_sequence=["#60a5fa", "#34d399", "#fbbf24", "#f87171"],
            zoom=9.5,
            center={"lat": lat, "lon": long},
            mapbox_style="carto-positron",
            height=500,
        )

        # Add target property star marker
        fig_map.add_trace(
            go.Scattermapbox(
                lat=[lat],
                lon=[long],
                mode="markers+text",
                marker=go.scattermapbox.Marker(size=18, color="#dc2626", symbol="circle"),
                text=["Subject Property"],
                textposition="top right",
                name="Subject Property",
            )
        )
        fig_map.update_layout(margin=dict(l=0, r=0, t=10, b=0), legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.02))
        st.plotly_chart(fig_map, use_container_width=True)

    # ----------------------------------------------------
    # TAB 3: AI Explainability (SHAP)
    # ----------------------------------------------------
    with tab_explain:
        st.subheader("Explainable AI: Local & Global Valuation Drivers")
        st.markdown(
            "Every prediction in EstateIQ Pro is backed by **SHAP (SHapley Additive exPlanations)**, "
            "showing exactly how each feature drives the valuation above or below base market expectation."
        )

        exp_col1, exp_col2 = st.columns([1.1, 0.9])
        with exp_col1:
            st.markdown("##### 🔍 Per-Listing Attribution Waterfall")
            render_shap_waterfall(features, predictor)

        with exp_col2:
            st.markdown("##### 🌐 Global Feature Importance (Trained Model)")
            if (ARTIFACTS_DIR / "shap_summary.png").exists():
                st.image(str(ARTIFACTS_DIR / "shap_summary.png"), use_container_width=True)
            elif (ARTIFACTS_DIR / "feature_importance.png").exists():
                st.image(str(ARTIFACTS_DIR / "feature_importance.png"), use_container_width=True)

    # ----------------------------------------------------
    # TAB 4: Model Benchmarks & Metrics
    # ----------------------------------------------------
    with tab_model:
        st.subheader("Model Performance & Empirical Benchmarks")
        st.markdown(
            "To ensure production reliability, EstateIQ Pro benchmarks linear baselines against advanced gradient boosted ensembles, "
            "followed by Bayesian Hyperparameter Optimization with Optuna on 5-Fold Cross Validation."
        )

        if meta.get("baseline_metrics") and meta.get("test_metrics"):
            b = meta["baseline_metrics"]
            t = meta["test_metrics"]

            comp_data = [
                {"Algorithm": "Ridge Regression (Baseline)", "R² Score": f"{b.get('Ridge', {}).get('r2', 0):.4f}", "MAPE (%)": f"{b.get('Ridge', {}).get('mape', 0):.2f}%", "MAE ($)": f"${b.get('Ridge', {}).get('mae', 0):,.0f}"},
                {"Algorithm": "Random Forest Regressor", "R² Score": f"{b.get('RandomForest', {}).get('r2', 0):.4f}", "MAPE (%)": f"{b.get('RandomForest', {}).get('mape', 0):.2f}%", "MAE ($)": f"${b.get('RandomForest', {}).get('mae', 0):,.0f}"},
                {"Algorithm": "XGBoost Regressor", "R² Score": f"{b.get('XGBoost', {}).get('r2', 0):.4f}", "MAPE (%)": f"{b.get('XGBoost', {}).get('mape', 0):.2f}%", "MAE ($)": f"${b.get('XGBoost', {}).get('mae', 0):,.0f}"},
                {"Algorithm": "LightGBM (Default)", "R² Score": f"{b.get('LightGBM (default)', {}).get('r2', 0):.4f}", "MAPE (%)": f"{b.get('LightGBM (default)', {}).get('mape', 0):.2f}%", "MAE ($)": f"${b.get('LightGBM (default)', {}).get('mae', 0):,.0f}"},
                {"Algorithm": "🏆 LightGBM (Optuna Tuned - Production)", "R² Score": f"{t.get('r2', 0):.4f}", "MAPE (%)": f"{t.get('mape', 0):.2f}%", "MAE ($)": f"${t.get('mae', 0):,.0f}"},
            ]
            st.dataframe(pd.DataFrame(comp_data), hide_index=True, use_container_width=True)

        col_art1, col_art2 = st.columns(2)
        with col_art1:
            if (ARTIFACTS_DIR / "actual_vs_predicted.png").exists():
                st.markdown("##### Actual vs Predicted (Holdout Test Set)")
                st.image(str(ARTIFACTS_DIR / "actual_vs_predicted.png"), use_container_width=True)

        with col_art2:
            if (ARTIFACTS_DIR / "feature_importance.png").exists():
                st.markdown("##### Top 20 Feature Importances")
                st.image(str(ARTIFACTS_DIR / "feature_importance.png"), use_container_width=True)

    # ----------------------------------------------------
    # TAB 5: Valuation Report Export
    # ----------------------------------------------------
    with tab_report:
        st.subheader("Export Formal Property Valuation Report")
        st.write("Generate a client-ready, styled HTML valuation report including property details, confidence bounds, and market benchmarks.")

        report_html = build_report_html(result, features)
        st.download_button(
            label="📥 Download Valuation Report (.html)",
            data=report_html,
            file_name=f"estateiq_valuation_{zipcode}_{sqft_living}sqft.html",
            mime="text/html",
            type="primary",
            use_container_width=True,
        )


if __name__ == "__main__":
    main()

