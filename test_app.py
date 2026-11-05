"""
EstateIQ — Automated Test Suite
Verifies data loading, regression training, evaluation metrics, and artifact export.
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from modules.data_loader import load_default_dataset, compute_dataset_summary
from modules.regression import train_valuation_model


def run_tests():
    print("=" * 65)
    print("ESTATEIQ SYSTEM VERIFICATION")
    print("=" * 65)

    # 1. Dataset Loading
    df = load_default_dataset("data/train.csv")
    assert not df.empty, "Failed to load default dataset"
    assert df.shape == (1460, 81), f"Unexpected shape {df.shape}"
    print(f"[PASS] 1. Data Loader: Ingested {df.shape[0]:,} rows x {df.shape[1]} cols")

    # 2. KPI Summary Computation
    summary = compute_dataset_summary(df, target_col="SalePrice")
    assert summary["total_records"] == 1460
    assert np.isclose(summary["mean_price"], 180921.20, atol=1.0)
    assert np.isclose(summary["median_price"], 163000.0, atol=1.0)
    print(f"[PASS] 2. Summary KPI: Mean ${summary['mean_price']:,.2f}, Median ${summary['median_price']:,.2f}")

    # 3. Model Training & OLS Regression
    bundle = train_valuation_model(df, feature_col="GrLivArea", target_col="SalePrice")
    assert bundle, "Failed to train valuation model"
    m = bundle["slope_m"]
    c = bundle["intercept_c"]
    assert np.isclose(m, 102.4890, atol=1e-2), f"Unexpected slope {m}"
    assert np.isclose(c, 24899.7482, atol=1.0), f"Unexpected intercept {c}"
    print(f"[PASS] 3. Regression Parameters: Slope = ${m:.2f}/sq ft, Intercept = ${c:,.2f}")

    # 4. Evaluation Metrics
    y_test = bundle["y_test"]
    y_pred_test = bundle["y_pred_test"]
    mae = mean_absolute_error(y_test, y_pred_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2 = r2_score(y_test, y_pred_test)

    assert np.isclose(mae, 38341.20, atol=1.0), f"Unexpected MAE: {mae}"
    assert np.isclose(rmse, 58471.76, atol=1.0), f"Unexpected RMSE: {rmse}"
    assert np.isclose(r2, 0.5543, atol=1e-3), f"Unexpected R2: {r2}"
    print(f"[PASS] 4. Evaluation: MAE = ${mae:,.2f}, RMSE = ${rmse:,.2f}, R2 = {r2:.4f}")

    # 5. Valuation Calculation
    test_sqft = 1800
    pred_1800 = float(bundle["model"].predict(pd.DataFrame([[test_sqft]], columns=["GrLivArea"]))[0])
    expected = (m * test_sqft) + c
    assert np.isclose(pred_1800, expected), "Valuation formula mismatch"
    print(f"[PASS] 5. Valuation: 1,800 sq ft -> ${pred_1800:,.2f}")

    # 6. Artifact Serialization (models/regression_model.pkl)
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "regression_model.pkl")
    save_data = {
        "model": bundle["model"],
        "slope_m": m,
        "intercept_c": c,
        "feature_name": "GrLivArea",
        "target_name": "SalePrice"
    }
    joblib.dump(save_data, model_path)
    assert os.path.exists(model_path)
    reloaded = joblib.load(model_path)
    reloaded_pred = float(reloaded["model"].predict(pd.DataFrame([[test_sqft]], columns=["GrLivArea"]))[0])
    assert np.isclose(reloaded_pred, pred_1800)
    print(f"[PASS] 6. Model Persistence: Serialized and reloaded successfully.")

    # 7. Output Export (outputs/predictions.csv)
    os.makedirs("outputs", exist_ok=True)
    preds_path = os.path.join("outputs", "predictions.csv")
    preds_df = pd.DataFrame({
        "GrLivArea": bundle["X_test"]["GrLivArea"].values,
        "ActualPrice": y_test.values,
        "PredictedPrice": y_pred_test,
        "AbsoluteError": np.abs(y_test.values - y_pred_test)
    })
    preds_df.to_csv(preds_path, index=True)
    assert os.path.exists(preds_path)
    assert len(pd.read_csv(preds_path)) == 292
    print(f"[PASS] 7. Export: {preds_path} written ({len(preds_df)} rows)")

    # 8. Clean Module Imports
    import modules.data_loader
    import modules.preprocessing
    import modules.analysis
    import modules.regression
    import modules.valuation
    import modules.evaluation
    print(f"[PASS] 8. Architecture: All 6 product modules imported cleanly.")

    print("=" * 65)
    print("ALL 8 ESTATEIQ VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 65)


if __name__ == "__main__":
    run_tests()
