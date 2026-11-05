"""
EstateIQ — Data Loader Module
Handles ingestion of local datasets and user-uploaded CSV files.
"""

import os
import streamlit as st
import pandas as pd
import numpy as np


@st.cache_data
def load_default_dataset(filepath: str = "data/train.csv") -> pd.DataFrame:
    """
    Loads the default Ames Housing dataset from the local path.
    Returns a pandas DataFrame.
    """
    if os.path.exists(filepath):
        return pd.read_csv(filepath)
    return pd.DataFrame()


def load_uploaded_dataset(uploaded_file) -> pd.DataFrame:
    """
    Reads a user-provided CSV file and returns a DataFrame.
    """
    if uploaded_file is not None:
        try:
            return pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(f"Error parsing CSV file: {e}")
            return pd.DataFrame()
    return pd.DataFrame()


def compute_dataset_summary(df: pd.DataFrame, target_col: str = "SalePrice") -> dict:
    """
    Computes summary KPI statistics for a given property dataset.
    """
    if df.empty:
        return {}

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    has_target = target_col in df.columns
    target_series = df[target_col].dropna() if has_target else pd.Series(dtype=float)

    return {
        "total_records": len(df),
        "total_features": df.shape[1],
        "numerical_features_count": len(num_cols),
        "categorical_features_count": len(cat_cols),
        "numerical_columns": num_cols,
        "categorical_columns": cat_cols,
        "target_col": target_col,
        "has_target": has_target,
        "mean_price": float(target_series.mean()) if not target_series.empty else 0.0,
        "median_price": float(target_series.median()) if not target_series.empty else 0.0,
        "min_price": float(target_series.min()) if not target_series.empty else 0.0,
        "max_price": float(target_series.max()) if not target_series.empty else 0.0,
        "std_price": float(target_series.std()) if not target_series.empty else 0.0
    }

