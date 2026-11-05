"""
EstateIQ — Market Analysis Module
Visualizes property distributions, bivariate relationships, outlier structures, and feature correlations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def render_market_analysis(df: pd.DataFrame):
    """
    Renders statistical charts and feature comparisons for market exploration.
    """
    st.markdown('<div class="page-title">Market Analysis</div>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Examine pricing distributions, square footage relationships, and correlations across residential attributes.</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if df.empty or "SalePrice" not in df.columns or "GrLivArea" not in df.columns:
        st.warning("Analysis requires a dataset containing 'SalePrice' and 'GrLivArea'.")
        return

    # Global chart styling
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.edgecolor"] = "#E5E7E1"
    plt.rcParams["axes.linewidth"] = 0.8

    corr_val = float(df["GrLivArea"].corr(df["SalePrice"]))
    skew_price = float(df["SalePrice"].skew())
    skew_area = float(df["GrLivArea"].skew())

    st.markdown(f"""
    <div class="metrics-grid">
        <div class="metric-box">
            <div class="metric-name">Pearson Correlation (r)</div>
            <div class="metric-val">+{corr_val:.2f}</div>
            <div class="metric-sub">Living Area vs Sale Price</div>
        </div>
        <div class="metric-box">
            <div class="metric-name">SalePrice Skewness</div>
            <div class="metric-val">{skew_price:.2f}</div>
            <div class="metric-sub">Right-skewed pricing distribution</div>
        </div>
        <div class="metric-box">
            <div class="metric-name">Living Area Skewness</div>
            <div class="metric-val">{skew_area:.2f}</div>
            <div class="metric-sub">Mild right-skew</div>
        </div>
        <div class="metric-box">
            <div class="metric-name">Coefficient of Determination</div>
            <div class="metric-val">{(corr_val**2):.2f}</div>
            <div class="metric-sub">Shared bivariate variance</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_scatter, tab_price_dist, tab_box, tab_corr, tab_custom = st.tabs([
        "Price vs. Living Area",
        "Price Distribution",
        "Price Outlier Analysis",
        "Feature Correlation Heatmap",
        "Interactive Feature Explorer"
    ])

    # 1. Price vs Living Area Scatter with Linear Fit
    with tab_scatter:
        st.markdown("**Relationship Between Above-Ground Living Area and Sale Price:**")
        fig, ax = plt.subplots(figsize=(9, 4.8), facecolor="#FFFFFF")
        ax.set_facecolor("#FAFAFA")
        sns.regplot(
            data=df, x="GrLivArea", y="SalePrice", ax=ax,
            scatter_kws={"color": "#244B3C", "alpha": 0.4, "s": 30},
            line_kws={"color": "#C08A3E", "linewidth": 2, "label": f"OLS Trendline (r = +{corr_val:.2f})"}
        )
        ax.set_title("Sale Price vs. Above-Ground Living Area", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Above-Ground Living Area (sq ft)", fontsize=10, fontweight="600")
        ax.set_ylabel("Sale Price ($)", fontsize=10, fontweight="600")
        ax.yaxis.set_major_formatter("${x:,.0f}")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(frameon=True, facecolor="#FFFFFF")
        st.pyplot(fig)
        plt.close(fig)

        st.markdown(f"""
        <div class="notice-text">
            <strong>Observation:</strong> There is a strong positive correlation ($r = +{corr_val:.2f}$) between finished above-ground square footage and transaction price. 
            Properties up to 2,500 sq ft track tightly with the linear fit. Above 2,500 sq ft, variance widens substantially due to luxury finishes, lot premium, and condition.
        </div>
        """, unsafe_allow_html=True)

    # 2. Price Distribution Histogram
    with tab_price_dist:
        st.markdown("**Distribution of Recorded Property Sale Prices:**")
        fig, ax = plt.subplots(figsize=(9, 4.4), facecolor="#FFFFFF")
        ax.set_facecolor("#FAFAFA")
        sns.histplot(df["SalePrice"], kde=True, color="#244B3C", bins=35, edgecolor="#1A362B", alpha=0.6, ax=ax)
        
        mean_p = df["SalePrice"].mean()
        med_p = df["SalePrice"].median()
        ax.axvline(mean_p, color="#C08A3E", linestyle="--", linewidth=1.6, label=f"Mean: ${mean_p:,.0f}")
        ax.axvline(med_p, color="#244B3C", linestyle="-", linewidth=1.6, label=f"Median: ${med_p:,.0f}")
        
        ax.set_title("Sale Price Distribution and Central Tendency", fontsize=12, fontweight="bold", pad=12)
        ax.set_xlabel("Sale Price ($)", fontsize=10, fontweight="600")
        ax.set_ylabel("Number of Properties", fontsize=10, fontweight="600")
        ax.xaxis.set_major_formatter("${x:,.0f}")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(frameon=True, facecolor="#FFFFFF")
        st.pyplot(fig)
        plt.close(fig)

        st.caption(f"Right-skewed market distribution (skewness = {skew_price:.2f}). Fifty percent of properties sold below ${med_p:,.0f}, while high-value transactions extend beyond $500,000.")

    # 3. Outlier Analysis via Box Plot
    with tab_box:
        st.markdown("**Sale Price Quartile Dispersion & Outlier Bounds:**")
        fig, ax = plt.subplots(figsize=(9, 3.4), facecolor="#FFFFFF")
        ax.set_facecolor("#FAFAFA")
        sns.boxplot(x=df["SalePrice"], ax=ax, color="#829B83", flierprops={"marker": "o", "color": "#244B3C", "alpha": 0.45, "markersize": 4})
        ax.set_title("Sale Price Box Plot (Tukey 1.5 IQR)", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel("Sale Price ($)", fontsize=10, fontweight="600")
        ax.xaxis.set_major_formatter("${x:,.0f}")
        ax.grid(True, linestyle=":", alpha=0.5)
        st.pyplot(fig)
        plt.close(fig)

        q1 = df["SalePrice"].quantile(0.25)
        q3 = df["SalePrice"].quantile(0.75)
        iqr = q3 - q1
        outlier_threshold = q3 + 1.5 * iqr
        outlier_count = int((df["SalePrice"] > outlier_threshold).sum())

        st.caption(f"25th Percentile: ${q1:,.0f} | Median: ${df['SalePrice'].median():,.0f} | 75th Percentile: ${q3:,.0f} | IQR: ${iqr:,.0f}. Properties priced above ${outlier_threshold:,.0f} ({outlier_count} listings) represent statistical upper outliers.")

    # 4. Correlation Heatmap
    with tab_corr:
        st.markdown("**Correlation Matrix of Numerical Property Attributes:**")
        candidate_cols = [
            "SalePrice", "GrLivArea", "OverallQual", "TotalBsmtSF", 
            "1stFlrSF", "GarageCars", "GarageArea", "YearBuilt", "FullBath", "TotRmsAbvGrd"
        ]
        available_cols = [c for c in candidate_cols if c in df.columns]
        corr_matrix = df[available_cols].corr()

        fig, ax = plt.subplots(figsize=(9, 5.5), facecolor="#FFFFFF")
        sns.heatmap(
            corr_matrix, annot=True, fmt=".2f", cmap="Greens", vmin=0.2, vmax=1.0,
            square=True, linewidths=0.5, linecolor="#FFFFFF", ax=ax, cbar_kws={"shrink": 0.8}
        )
        ax.set_title("Correlation Heatmap (Top Physical Drivers)", fontsize=12, fontweight="bold", pad=12)
        plt.xticks(rotation=45, ha="right", fontsize=9)
        plt.yticks(rotation=0, fontsize=9)
        st.pyplot(fig)
        plt.close(fig)

        st.caption("GrLivArea (0.71) is the strongest continuous dimensional predictor of sale price, closely following OverallQual (0.79).")

    # 5. Interactive Feature Explorer
    with tab_custom:
        st.markdown("**Compare Any Numerical Feature Against Sale Price:**")
        numeric_options = [c for c in df.select_dtypes(include=[np.number]).columns if c != "SalePrice" and c != "Id"]
        
        default_idx = numeric_options.index("TotalBsmtSF") if "TotalBsmtSF" in numeric_options else 0
        selected_feature = st.selectbox("Select Numerical Feature:", numeric_options, index=default_idx)

        c_corr = float(df[selected_feature].corr(df["SalePrice"]))

        fig, ax = plt.subplots(figsize=(9, 4.4), facecolor="#FFFFFF")
        ax.set_facecolor("#FAFAFA")
        sns.regplot(
            data=df, x=selected_feature, y="SalePrice", ax=ax,
            scatter_kws={"color": "#244B3C", "alpha": 0.35, "s": 25},
            line_kws={"color": "#C08A3E", "linewidth": 2, "label": f"Fit (r = +{c_corr:.2f})"}
        )
        ax.set_title(f"Sale Price vs. {selected_feature}", fontsize=12, fontweight="bold", pad=10)
        ax.set_xlabel(selected_feature, fontsize=10, fontweight="600")
        ax.set_ylabel("Sale Price ($)", fontsize=10, fontweight="600")
        ax.yaxis.set_major_formatter("${x:,.0f}")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend(frameon=True, facecolor="#FFFFFF")
        st.pyplot(fig)
        plt.close(fig)
        st.caption(f"Correlation between {selected_feature} and SalePrice: r = +{c_corr:.4f}")

