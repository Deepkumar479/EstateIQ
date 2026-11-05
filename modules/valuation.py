"""
EstateIQ — Valuation Module
Combines model equation derivation, interactive property estimation, and artifact persistence.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from modules.regression import render_valuation_engine_tab


def render_valuation(reg_bundle: dict, df: pd.DataFrame):
    """
    Renders the Valuation section: Model parameters, interactive calculator, and model export.
    """
    st.markdown('<div class="page-title">Valuation</div>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Calculate property valuations using linear regression parameters fitted on historical market transactions.</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if not reg_bundle:
        st.warning("Valuation model is not currently available.")
        return

    model = reg_bundle["model"]
    slope_m = reg_bundle["slope_m"]
    intercept_c = reg_bundle["intercept_c"]
    feature_col = reg_bundle["feature_col"]

    min_area = int(df[feature_col].min()) if feature_col in df.columns else 334
    max_area = int(df[feature_col].max()) if feature_col in df.columns else 5642

    st.markdown("### 1. Property Value Estimator")

    col_in, col_out = st.columns([1, 1])

    with col_in:
        user_sqft = st.number_input(
            "Enter Above-Ground Living Area (sq ft):",
            min_value=100,
            max_value=10000,
            value=1800,
            step=50,
            help=f"Observed training range: {min_area:,} to {max_area:,} sq ft"
        )

        st.markdown("**Common Formats:**")
        p1, p2, p3 = st.columns(3)
        with p1:
            if st.button("950 sq ft"):
                user_sqft = 950
        with p2:
            if st.button("1,800 sq ft"):
                user_sqft = 1800
        with p3:
            if st.button("2,800 sq ft"):
                user_sqft = 2800

    # Calculate valuation: Y = m*X + c
    pred_val = float(model.predict(pd.DataFrame([[user_sqft]], columns=[feature_col]))[0])

    with col_out:
        st.markdown(f"""
        <div class="valuation-result">
            <div class="val-label">Estimated Valuation</div>
            <div class="val-amount">${pred_val:,.2f}</div>
            <div class="val-meta">Area: <strong>{user_sqft:,} sq ft</strong> &bull; Rate: <strong>${slope_m:.2f} / sq ft</strong></div>
        </div>
        """, unsafe_allow_html=True)

        if user_sqft < min_area:
            st.markdown(f'<div class="warning-text">Input {user_sqft:,} sq ft is smaller than the minimum observed property in the dataset ({min_area:,} sq ft). The estimate involves lower-bound extrapolation.</div>', unsafe_allow_html=True)
        elif user_sqft > max_area:
            st.markdown(f'<div class="warning-text">Input {user_sqft:,} sq ft exceeds the maximum observed property in the dataset ({max_area:,} sq ft). Large luxury homes often exhibit non-linear pricing variations.</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div class="code-calc-box">
        Estimated Price = (${slope_m:.2f} &times; {user_sqft:,} sq ft) + ${intercept_c:,.2f} = <strong>${pred_val:,.2f}</strong>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="notice-text">
        <strong>Model Assumptions & Valuation Disclaimer:</strong><br>
        This estimate represents a statistical projection calculated by a univariate linear regression model trained on recorded residential property sales. 
        It assumes average build quality, typical lot size, standard condition, and average neighborhood desirability. 
        Real estate valuation in practice is influenced by location, neighborhood amenities, construction materials, architectural condition, and current mortgage interest rates. This output is an analytical estimate and does not constitute a certified appraisal or guaranteed market quotation.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 2. Regression Engine & Parameters")
    render_valuation_engine_tab(reg_bundle)

    st.markdown("---")
    st.markdown("### 3. Model Persistence & Export")

    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    model_path = os.path.join("models", "regression_model.pkl")
    preds_path = os.path.join("outputs", "predictions.csv")

    save_bundle = {
        "model": model,
        "feature_name": feature_col,
        "target_name": reg_bundle["target_col"],
        "slope_m": slope_m,
        "intercept_c": intercept_c,
        "equation": reg_bundle["equation_str"],
        "train_samples": reg_bundle["train_size"],
        "test_samples": reg_bundle["test_size"],
        "train_r2": reg_bundle["train_r2"],
        "test_r2": reg_bundle["test_r2"]
    }
    joblib.dump(save_bundle, model_path)

    # Predictions CSV
    y_test = reg_bundle["y_test"]
    y_pred_test = reg_bundle["y_pred_test"]
    X_test = reg_bundle["X_test"]
    
    preds_df = pd.DataFrame({
        feature_col: X_test[feature_col].values,
        "ActualPrice": y_test.values,
        "PredictedPrice": y_pred_test,
        "AbsoluteError": np.abs(y_test.values - y_pred_test),
        "PercentageError": (np.abs(y_test.values - y_pred_test) / y_test.values) * 100
    }, index=y_test.index)
    preds_df.to_csv(preds_path, index=True)

    c_dl1, c_dl2 = st.columns(2)
    with c_dl1:
        if os.path.exists(model_path):
            with open(model_path, "rb") as f:
                st.download_button(
                    label="Download Fitted Model (.pkl)",
                    data=f,
                    file_name="regression_model.pkl",
                    mime="application/octet-stream"
                )
    with c_dl2:
        if os.path.exists(preds_path):
            with open(preds_path, "r", encoding="utf-8") as f:
                st.download_button(
                    label="Export Test Predictions (.csv)",
                    data=f.read(),
                    file_name="predictions.csv",
                    mime="text/csv"
                )

