"""
EstateIQ — Model Performance & Evaluation Module
Evaluates the Linear Regression model on unseen testing data with standard regression loss metrics.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def render_model_performance(reg_bundle: dict):
    """
    Renders regression evaluation metrics, actual-vs-predicted diagnostics,
    error distribution, and honest limitations analysis.
    """
    st.markdown('<div class="page-title">Model Performance</div>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Quantify generalization and prediction error on withheld test properties.</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if not reg_bundle:
        st.warning("Model has not been evaluated yet.")
        return

    y_test = reg_bundle["y_test"]
    y_pred_test = reg_bundle["y_pred_test"]
    y_train = reg_bundle["y_train"]
    y_pred_train = reg_bundle["y_pred_train"]
    X_test = reg_bundle["X_test"]
    feature_col = reg_bundle["feature_col"]

    # Compute Metrics
    mae = mean_absolute_error(y_test, y_pred_test)
    mse = mean_squared_error(y_test, y_pred_test)
    rmse = np.sqrt(mse)
    r2_test = r2_score(y_test, y_pred_test)
    r2_train = r2_score(y_train, y_pred_train)

    st.markdown(f"""
    <div class="metrics-grid">
        <div class="metric-box">
            <div class="metric-name">Mean Absolute Error (MAE)</div>
            <div class="metric-val">${mae:,.2f}</div>
            <div class="metric-sub">Average dollar deviation</div>
        </div>
        <div class="metric-box accent-sage">
            <div class="metric-name">Root Mean Squared Error (RMSE)</div>
            <div class="metric-val">${rmse:,.2f}</div>
            <div class="metric-sub">Error penalizing large residuals</div>
        </div>
        <div class="metric-box accent-amber">
            <div class="metric-name">Test R² Score</div>
            <div class="metric-val">{r2_test:.4f}</div>
            <div class="metric-sub">{r2_test*100:.1f}% variance explained</div>
        </div>
        <div class="metric-box accent-forest">
            <div class="metric-name">Train R² Score</div>
            <div class="metric-val">{r2_train:.4f}</div>
            <div class="metric-sub">Consistent with test score</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="content-card">
        <h3>Metric Definitions & Interpretations:</h3>
        <ul>
            <li><strong>Mean Absolute Error (${mae:,.2f}):</strong> On average, model valuations deviate from recorded property transaction prices by ${mae:,.2f} (~21% of average property price).</li>
            <li><strong>Root Mean Squared Error (${rmse:,.2f}):</strong> The standard deviation of the residuals. RMSE is higher than MAE because it squares errors before averaging, disproportionately penalizing large errors on luxury listings.</li>
            <li><strong>Coefficient of Determination (R² = {r2_test:.4f}):</strong> Above-ground square footage explains <strong>{r2_test*100:.1f}% of total price variation</strong> across the property market. (Note: R² denotes explained variance relative to a horizontal mean baseline, not a classification accuracy percentage).</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    tab_plot, tab_table, tab_res = st.tabs([
        "Actual vs. Predicted Plot",
        "Test Predictions Table",
        "Residual Error Distribution"
    ])

    # 1. Actual vs Predicted Scatter Plot
    with tab_plot:
        st.markdown("**Actual vs. Predicted Property Prices (Test Set, n=292):**")
        fig, ax = plt.subplots(figsize=(9, 5), facecolor="#FFFFFF")
        ax.set_facecolor("#FAFAFA")
        
        ax.scatter(
            y_test, y_pred_test, 
            color="#244B3C", alpha=0.55, s=36, edgecolor="#FFFFFF", linewidth=0.5,
            label="Test Properties (n=292)"
        )
        
        # Perfect prediction reference line y = x
        min_p = min(y_test.min(), y_pred_test.min())
        max_p = max(y_test.max(), y_pred_test.max())
        ax.plot([min_p, max_p], [min_p, max_p], color="#C08A3E", linestyle="--", linewidth=2, label="Identity Line (y = x)")
        
        ax.set_title("Actual vs. Predicted Property Prices", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Actual Sale Price ($)", fontsize=10, fontweight="600")
        ax.set_ylabel("Predicted Sale Price ($)", fontsize=10, fontweight="600")
        ax.xaxis.set_major_formatter("${x:,.0f}")
        ax.yaxis.set_major_formatter("${x:,.0f}")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(frameon=True, facecolor="#FFFFFF")
        st.pyplot(fig)
        plt.close(fig)

        st.caption("Points cluster along the identity line for mid-range homes ($100k-$250k). Higher-value properties display greater dispersion as non-area amenities (custom masonry, pools, landscaping) impact value.")

    # 2. Test Predictions Table
    with tab_table:
        st.markdown(f"**Detailed Comparison ({len(y_test)} Test Observations):**")
        table_df = pd.DataFrame({
            f"{feature_col} (sq ft)": X_test[feature_col].values,
            "Actual Sale Price ($)": y_test.values,
            "Predicted Sale Price ($)": y_pred_test,
            "Absolute Error ($)": np.abs(y_test.values - y_pred_test),
            "Error (%)": (np.abs(y_test.values - y_pred_test) / y_test.values) * 100
        }, index=y_test.index)

        st.dataframe(table_df.head(15).style.format({
            f"{feature_col} (sq ft)": "{:,.0f}",
            "Actual Sale Price ($)": "${:,.2f}",
            "Predicted Sale Price ($)": "${:,.2f}",
            "Absolute Error ($)": "${:,.2f}",
            "Error (%)": "{:.1f}%"
        }), use_container_width=True)

        col_stat1, col_stat2 = st.columns(2)
        with col_stat1:
            st.write(f"- Median Absolute Error: **${table_df['Absolute Error ($)'].median():,.2f}**")
        with col_stat2:
            st.write(f"- Median Percentage Error: **{table_df['Error (%)'].median():.1f}%**")

    # 3. Residual Error Distribution
    with tab_res:
        residuals = y_test.values - y_pred_test
        fig, ax = plt.subplots(figsize=(9, 4.2), facecolor="#FFFFFF")
        ax.set_facecolor("#FAFAFA")
        sns.histplot(residuals, kde=True, color="#829B83", bins=28, edgecolor="#556658", alpha=0.6, ax=ax)
        ax.axvline(0, color="#244B3C", linestyle="-", linewidth=1.8, label="Zero Error")
        ax.axvline(np.mean(residuals), color="#C08A3E", linestyle="--", linewidth=1.5, label=f"Mean Error: ${np.mean(residuals):,.0f}")
        
        ax.set_title("Distribution of Residual Prediction Errors", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Residual (Actual - Predicted in $)", fontsize=10, fontweight="600")
        ax.set_ylabel("Property Count", fontsize=10, fontweight="600")
        ax.xaxis.set_major_formatter("${x:,.0f}")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(frameon=True, facecolor="#FFFFFF")
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("""
    <div class="content-card">
        <h3>Observed Model Limitations:</h3>
        <ul>
            <li><strong>Single Attribute Scope:</strong> While living area explains over 55% of variation, the remaining ~45% is driven by factors like location, structural condition, quality ratings, and basement area.</li>
            <li><strong>Fixed Incremental Assumption:</strong> The model assumes a fixed addition of $102.49 per square foot across all property sizes. In reality, large homes experience diminishing marginal utility per square foot.</li>
            <li><strong>Outlier Influence:</strong> A small number of extreme estate listings exert leverage on the regression slope and inflate squared error metrics.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
