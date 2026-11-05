"""
EstateIQ — Linear Regression Engine
Trains and validates Ordinary Least Squares (OLS) regression models for property pricing.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


def train_valuation_model(df: pd.DataFrame, feature_col: str = "GrLivArea", target_col: str = "SalePrice") -> dict:
    """
    Executes an 80:20 train-test split, fits LinearRegression on the training partition,
    and returns model parameters, splits, and metadata.
    """
    if df.empty or feature_col not in df.columns or target_col not in df.columns:
        return {}

    # 1. Clean missing or invalid entries for modeling variables
    valid_mask = (df[feature_col].notnull()) & (df[target_col].notnull()) & (df[feature_col] > 0) & (df[target_col] > 0)
    clean_subset = df[valid_mask].copy()

    X = clean_subset[[feature_col]]
    y = clean_subset[target_col]

    # 2. Strict 80:20 train-test split with fixed seed
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # 3. Fit Linear Regression strictly on training data (no data leakage)
    model = LinearRegression()
    model.fit(X_train, y_train)

    slope_m = float(model.coef_[0])
    intercept_c = float(model.intercept_)

    # 4. Generate predictions
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)

    return {
        "model": model,
        "feature_col": feature_col,
        "target_col": target_col,
        "slope_m": slope_m,
        "intercept_c": intercept_c,
        "equation_str": f"SalePrice = {slope_m:.2f} * {feature_col} + ${intercept_c:,.2f}",
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "y_pred_test": y_pred_test,
        "y_pred_train": y_pred_train,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "train_r2": float(model.score(X_train, y_train)),
        "test_r2": float(model.score(X_test, y_test))
    }


def render_valuation_engine_tab(reg_bundle: dict):
    """
    Renders the model architecture, parameters, and regression equation.
    """
    if not reg_bundle:
        st.warning("Valuation model is not trained yet.")
        return

    m = reg_bundle["slope_m"]
    c = reg_bundle["intercept_c"]
    feat = reg_bundle["feature_col"]

    st.markdown(f"""
    <div class="code-calc-box">
        <strong>Fitted Linear Regression Equation:</strong><br>
        SalePrice = ({m:.2f} &times; {feat}) + ${c:,.2f}
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"""
        <div class="content-card">
            <h3>Incremental Rate (Slope m): ${m:.2f} / sq ft</h3>
            <p>Every additional square foot of above-ground living space corresponds to an estimated price increase of approximately <strong>${m:.2f}</strong>, holding other factors constant.</p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class="content-card">
            <h3>Base Intercept (Constant c): ${c:,.2f}</h3>
            <p>The mathematical constant where the linear function crosses zero living area. Serves as an equation anchor rather than a standalone empty-lot valuation.</p>
        </div>
        """, unsafe_allow_html=True)

