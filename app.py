"""
EstateIQ — Property Insights & Valuation
Main application entry point.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from modules.data_loader import load_default_dataset, load_uploaded_dataset, compute_dataset_summary
from modules.preprocessing import render_property_data
from modules.analysis import render_market_analysis
from modules.regression import train_valuation_model
from modules.valuation import render_valuation
from modules.evaluation import render_model_performance


# Page Configuration
st.set_page_config(
    page_title="EstateIQ — Property Insights & Valuation",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)


# Load Custom Stylesheet
def load_stylesheet(css_file: str = "assets/style.css"):
    if os.path.exists(css_file):
        with open(css_file, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_stylesheet("assets/style.css")


# Initialize Session State
if "raw_df" not in st.session_state:
    raw_df = load_default_dataset()
    st.session_state.raw_df = raw_df.copy()
    st.session_state.clean_df = raw_df.copy()
    st.session_state.dataset_label = "Ames Housing Dataset (train.csv)"

if "reg_bundle" not in st.session_state:
    st.session_state.reg_bundle = None


# Ensure Model Trained
def get_or_train_model():
    if st.session_state.reg_bundle is None:
        clean_df = st.session_state.clean_df
        bundle = train_valuation_model(clean_df, feature_col="GrLivArea", target_col="SalePrice")
        st.session_state.reg_bundle = bundle
    return st.session_state.reg_bundle


# =============================================================================
# SIDEBAR
# =============================================================================
st.sidebar.markdown("""
<div style="padding: 10px 0 16px 0;">
    <div class="brand-wordmark">EstateIQ</div>
    <div class="brand-tagline">Property Insights & Valuation</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")

# Navigation
st.sidebar.markdown("**Navigation**")
active_section = st.sidebar.radio(
    "Go to:",
    [
        "Overview",
        "Property Data",
        "Market Analysis",
        "Valuation",
        "Model Performance"
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")

# Data Source Management
st.sidebar.markdown("**Data Source**")
source_mode = st.sidebar.radio(
    "Dataset Source:",
    ["Default Dataset", "Upload CSV"],
    label_visibility="collapsed"
)

if source_mode == "Upload CSV":
    uploaded = st.sidebar.file_uploader(
        "Upload property CSV",
        type=["csv"],
        help="Upload CSV with numerical property features"
    )
    if uploaded is not None:
        custom_df = load_uploaded_dataset(uploaded)
        if not custom_df.empty:
            st.session_state.raw_df = custom_df.copy()
            st.session_state.clean_df = custom_df.copy()
            st.session_state.dataset_label = uploaded.name
            st.session_state.reg_bundle = None
            st.sidebar.success(f"Loaded {len(custom_df):,} records")
else:
    if st.session_state.dataset_label != "Ames Housing Dataset (train.csv)":
        default_df = load_default_dataset()
        st.session_state.raw_df = default_df.copy()
        st.session_state.clean_df = default_df.copy()
        st.session_state.dataset_label = "Ames Housing Dataset (train.csv)"
        st.session_state.reg_bundle = None


# Current Data
current_raw = st.session_state.raw_df
if current_raw.empty:
    st.error("No dataset available. Please verify data/train.csv exists or upload a valid CSV file.")
    st.stop()


# Train model if needed
reg_bundle = get_or_train_model()


# =============================================================================
# SECTION ROUTING
# =============================================================================

if active_section == "Overview":
    st.markdown('<div class="page-title">Overview</div>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Residential property dataset metrics, summary distributions, and platform summary.</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    summary = compute_dataset_summary(current_raw, target_col="SalePrice")

    # Practical Metrics Grid (avoiding excessive cards)
    st.markdown(f"""
    <div class="metrics-grid">
        <div class="metric-box">
            <div class="metric-name">Property Records</div>
            <div class="metric-val">{summary.get('total_records', 0):,}</div>
            <div class="metric-sub">Total observations</div>
        </div>
        <div class="metric-box">
            <div class="metric-name">Features Available</div>
            <div class="metric-val">{summary.get('total_features', 0)}</div>
            <div class="metric-sub">{summary.get('numerical_features_count', 0)} num | {summary.get('categorical_features_count', 0)} cat</div>
        </div>
        <div class="metric-box">
            <div class="metric-name">Average Sale Price</div>
            <div class="metric-val">${summary.get('mean_price', 0):,.0f}</div>
            <div class="metric-sub">Recorded mean</div>
        </div>
        <div class="metric-box">
            <div class="metric-name">Median Sale Price</div>
            <div class="metric-val">${summary.get('median_price', 0):,.0f}</div>
            <div class="metric-sub">50th percentile</div>
        </div>
        <div class="metric-box">
            <div class="metric-name">Price Range</div>
            <div class="metric-val">${summary.get('min_price', 0):,.0f} - ${summary.get('max_price', 0):,.0f}</div>
            <div class="metric-sub">Min to max recorded</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Core Description & Workspace context
    col_desc1, col_desc2 = st.columns([1.5, 1])
    with col_desc1:
        st.markdown(f"""
        <div class="content-card">
            <h3>About EstateIQ</h3>
            <p>
                EstateIQ is a property analytics application that helps users explore residential property data, 
                understand the factors associated with property prices, and generate estimated property valuations using 
                statistical analysis and linear regression.
            </p>
            <p>
                <strong>Current Dataset:</strong> <code>{st.session_state.dataset_label}</code><br>
                <strong>Primary Predictive Feature:</strong> <code>GrLivArea</code> (Above-ground living area in sq ft)<br>
                <strong>Target Variable:</strong> <code>SalePrice</code> (Transaction price in USD)<br>
                <strong>Reference Source:</strong> <a href="https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/data" target="_blank" style="color: #244B3C; font-weight: 600;">Kaggle House Prices – Advanced Regression Techniques</a>
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_desc2:
        st.markdown("""
        <div class="content-card">
            <h3>Core Functionality</h3>
            <ul style="padding-left: 18px; margin: 0;">
                <li><strong>Property Data:</strong> Schema inspection, missing value imputation, duplicate removal.</li>
                <li><strong>Market Analysis:</strong> Bivariate relationships, distributions, and feature correlations.</li>
                <li><strong>Valuation:</strong> Parametric Linear Regression pricing and interactive estimator.</li>
                <li><strong>Model Performance:</strong> Generalization loss metrics (MAE, RMSE, R²) and residual plots.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # One Meaningful Visualisation: Distribution of Sale Prices
    st.markdown("### Market Price Distribution")
    fig, ax = plt.subplots(figsize=(10, 4.2), facecolor="#FFFFFF")
    ax.set_facecolor("#FAFAFA")
    sns.histplot(current_raw["SalePrice"], kde=True, color="#244B3C", bins=35, edgecolor="#1A362B", alpha=0.6, ax=ax)
    
    mean_p = summary.get("mean_price", 0)
    med_p = summary.get("median_price", 0)
    ax.axvline(mean_p, color="#C08A3E", linestyle="--", linewidth=1.6, label=f"Mean: ${mean_p:,.0f}")
    ax.axvline(med_p, color="#244B3C", linestyle="-", linewidth=1.6, label=f"Median: ${med_p:,.0f}")
    
    ax.set_title("Distribution of Recorded Residential Property Prices", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Recorded Sale Price ($)", fontsize=10, fontweight="600")
    ax.set_ylabel("Number of Properties", fontsize=10, fontweight="600")
    ax.xaxis.set_major_formatter("${x:,.0f}")
    ax.grid(True, linestyle=":", alpha=0.5)
    ax.legend(frameon=True, facecolor="#FFFFFF")
    st.pyplot(fig)
    plt.close(fig)

elif active_section == "Property Data":
    cleaned_df = render_property_data(current_raw)
    st.session_state.clean_df = cleaned_df

elif active_section == "Market Analysis":
    render_market_analysis(st.session_state.clean_df)

elif active_section == "Valuation":
    render_valuation(reg_bundle, st.session_state.clean_df)

elif active_section == "Model Performance":
    render_model_performance(reg_bundle)
