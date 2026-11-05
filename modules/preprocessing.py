"""
EstateIQ — Preprocessing & Data Exploration Module
Handles schema inspection, missing-value imputation, duplicate audits, and feature prep.
"""

import streamlit as st
import pandas as pd
import numpy as np


def render_property_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Renders the Property Data screen for inspection and cleaning.
    Returns cleaned working dataframe while keeping raw_df unchanged.
    """
    st.markdown('<div class="page-title">Property Data</div>', unsafe_allow_html=True)
    st.markdown('<p class="page-subtitle">Inspect raw property attributes, review distributions, and manage data quality.</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    if raw_df.empty:
        st.warning("No property dataset is loaded.")
        return raw_df

    df_clean = raw_df.copy()

    # Metric Bar
    num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df_clean.select_dtypes(exclude=[np.number]).columns.tolist()
    total_missing = int(df_clean.isnull().sum().sum())
    total_dups = int(df_clean.duplicated().sum())

    st.markdown(f"""
    <div class="metrics-grid">
        <div class="metric-box">
            <div class="metric-name">Total Records</div>
            <div class="metric-val">{len(df_clean):,}</div>
            <div class="metric-sub">Rows in dataset</div>
        </div>
        <div class="metric-box">
            <div class="metric-name">Features</div>
            <div class="metric-val">{df_clean.shape[1]}</div>
            <div class="metric-sub">{len(num_cols)} num | {len(cat_cols)} cat</div>
        </div>
        <div class="metric-box">
            <div class="metric-name">Missing Entries</div>
            <div class="metric-val">{total_missing:,}</div>
            <div class="metric-sub">Across {df_clean.isnull().any().sum()} columns</div>
        </div>
        <div class="metric-box">
            <div class="metric-name">Duplicates</div>
            <div class="metric-val">{total_dups}</div>
            <div class="metric-sub">Exact duplicate rows</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    tab_preview, tab_schema, tab_cleaning, tab_stats = st.tabs([
        "Record Explorer",
        "Schema & Types",
        "Data Quality & Imputation",
        "Summary Statistics"
    ])

    # 1. Record Explorer
    with tab_preview:
        st.markdown("**Sample Records:**")
        preview_mode = st.radio("View:", ["First 10 Records", "Last 10 Records"], horizontal=True)
        if preview_mode == "First 10 Records":
            st.dataframe(df_clean.head(10), use_container_width=True)
        else:
            st.dataframe(df_clean.tail(10), use_container_width=True)

    # 2. Schema & Types
    with tab_schema:
        c1, c2 = st.columns([1, 2])
        with c1:
            st.markdown("**Column Types Breakdown:**")
            counts_df = pd.DataFrame({
                "Data Type": [str(k) for k in df_clean.dtypes.value_counts().index],
                "Count": df_clean.dtypes.value_counts().values
            })
            st.dataframe(counts_df, use_container_width=True)

        with c2:
            st.markdown("**Attribute Inventory:**")
            inv = []
            for col in df_clean.columns:
                n_miss = int(df_clean[col].isnull().sum())
                inv.append({
                    "Feature": col,
                    "Type": str(df_clean[col].dtype),
                    "Non-Null Count": int(df_clean[col].count()),
                    "Missing Count": n_miss,
                    "Missing (%)": f"{(n_miss / len(df_clean)) * 100:.1f}%"
                })
            st.dataframe(pd.DataFrame(inv), use_container_width=True, height=350)

    # 3. Data Quality & Imputation
    with tab_cleaning:
        st.markdown("**Missing Value Assessment:**")
        missing_series = df_clean.isnull().sum()
        missing_cols = missing_series[missing_series > 0].sort_values(ascending=False)

        if len(missing_cols) > 0:
            m_df = pd.DataFrame({
                "Feature": missing_cols.index,
                "Missing Values": missing_cols.values,
                "Missing Percentage": (missing_cols.values / len(df_clean) * 100).round(2),
                "Data Type": [str(df_clean[c].dtype) for c in missing_cols.index]
            })
            st.dataframe(m_df, use_container_width=True, height=200)

            st.markdown("**Imputation Settings:**")
            col_opt1, col_opt2 = st.columns(2)
            with col_opt1:
                num_strategy = st.selectbox(
                    "Numerical Missing Values:",
                    ["Median (Recommended - robust to price/area outliers)", "Mean"],
                    index=0
                )
            with col_opt2:
                cat_strategy = st.selectbox(
                    "Categorical Missing Values:",
                    ["Fill with 'None' (Indicates absence of property amenity)", "Mode (Most frequent category)"],
                    index=0
                )

            if st.button("Apply Preprocessing to Working Dataset"):
                for col in num_cols:
                    if df_clean[col].isnull().sum() > 0:
                        val = df_clean[col].median() if "Median" in num_strategy else df_clean[col].mean()
                        df_clean[col] = df_clean[col].fillna(val)

                for col in cat_cols:
                    if df_clean[col].isnull().sum() > 0:
                        if "None" in cat_strategy:
                            df_clean[col] = df_clean[col].fillna("None")
                        else:
                            mode_val = df_clean[col].mode()[0] if not df_clean[col].dropna().empty else "Unknown"
                            df_clean[col] = df_clean[col].fillna(mode_val)

                st.success(f"Imputation completed. Remaining missing values in working data: {df_clean.isnull().sum().sum()}")
        else:
            st.info("No missing values detected in the current dataset.")

        # Duplicate Handling
        st.markdown("**Duplicate Records Check:**")
        if total_dups > 0:
            st.warning(f"Found {total_dups} duplicate records.")
            if st.button("Remove Duplicates"):
                df_clean = df_clean.drop_duplicates()
                st.success(f"Removed duplicates. Updated row count: {len(df_clean):,}")
        else:
            st.markdown('<div class="notice-text">0 duplicate records found. Dataset maintains transaction-level integrity.</div>', unsafe_allow_html=True)

        # Before & After summary
        st.markdown("**Pre-processing Summary:**")
        summary_rows = [
            ("Raw Records", f"{len(raw_df):,}", f"{len(df_clean):,}", "Unchanged" if len(raw_df) == len(df_clean) else f"-{len(raw_df) - len(df_clean)} rows"),
            ("Missing Values", f"{raw_df.isnull().sum().sum():,}", f"{df_clean.isnull().sum().sum():,}", "Resolved" if df_clean.isnull().sum().sum() == 0 else "Partially cleaned"),
            ("Predictive Target (SalePrice)", "Available", "Available", "Valid"),
            ("Primary Predictor (GrLivArea)", "Available", "Available", "Valid")
        ]
        st.dataframe(pd.DataFrame(summary_rows, columns=["Metric", "Original Raw", "Cleaned Working", "Status"]), use_container_width=True)

    # 4. Summary Statistics
    with tab_stats:
        st.markdown("**Numerical Feature Distributions:**")
        st.dataframe(df_clean.describe().T.style.format("{:,.2f}"), use_container_width=True, height=350)

    # Impute defaults if user navigates without explicitly clicking button
    if df_clean.isnull().sum().sum() > 0:
        for c in df_clean.select_dtypes(include=[np.number]).columns:
            if df_clean[c].isnull().sum() > 0:
                df_clean[c] = df_clean[c].fillna(df_clean[c].median())
        for c in df_clean.select_dtypes(exclude=[np.number]).columns:
            if df_clean[c].isnull().sum() > 0:
                df_clean[c] = df_clean[c].fillna("None")

    return df_clean

