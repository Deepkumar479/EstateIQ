# EstateIQ — Property Insights & Valuation

EstateIQ is a residential property analytics and valuation application. It enables users to explore property transaction data, examine the relationship between physical property characteristics and market prices, and generate statistical property valuations using Ordinary Least Squares (OLS) Linear Regression.

---

## 1. Product Identity

- **Product Name:** EstateIQ
- **Tagline:** Property Insights & Valuation
- **Core Technology:** Python, Streamlit, Scikit-learn, Pandas, NumPy, Matplotlib, Seaborn, Joblib
- **Primary Predictor:** `GrLivArea` (Above-ground finished living area in square feet)
- **Target Variable:** `SalePrice` (Recorded transaction price in USD)
- **Benchmark Dataset:** Kaggle Ames Housing Dataset (`train.csv`, 1,460 observations, 81 attributes)

---

## 2. Technical Architecture

```
estateiq/
│
├── app.py                     # Streamlit application entry point & section routing
├── test_app.py                # Automated 8-point regression test suite
├── requirements.txt           # Python dependency specification
├── README.md                  # System documentation
│
├── .streamlit/
│   ├── config.toml            # Application theme (Warm Ivory & Forest Green)
│   └── credentials.toml       # Headless runner credentials
│
├── assets/
│   └── style.css              # Minimal, product-focused stylesheet
│
├── data/
│   └── train.csv              # Benchmark residential property dataset
│
├── models/
│   └── regression_model.pkl   # Serialized model parameters and coefficients
│
├── modules/
│   ├── __init__.py            # Package initialization
│   ├── data_loader.py         # Ingestion, CSV upload, and KPI summaries
│   ├── preprocessing.py       # Data exploration, null audits, and imputation
│   ├── analysis.py            # Market distributions, scatter fits, and heatmaps
│   ├── regression.py          # OLS training engine and parameter derivations
│   ├── evaluation.py          # Model performance, error tables, and diagnostics
│   └── valuation.py           # Interactive property estimator and export tools
│
├── notebooks/
│   └── property_analysis.ipynb # Fully executed Jupyter Notebook
│
└── outputs/
    └── predictions.csv        # Exported predictions on test properties
```

---

## 3. Application Sections

### 1. Overview
High-level workspace summarizing active dataset dimensions (1,460 records, 81 attributes), average sale price (\$180,921), median sale price (\$163,000), minimum and maximum recorded prices, alongside the distribution of property prices.

### 2. Property Data
Data exploration interface providing head/tail record previews, attribute schema and data types, missing value audit across all columns, missing-value imputation (median for numerical, mode/'None' for categorical), duplicate checks, and before-and-after cleaning verification.

### 3. Market Analysis
Exploratory analytics including price distributions, scatter plots of living area vs. sale price with linear trendlines, Tukey 1.5 IQR outlier analysis, a multi-feature correlation heatmap, and an interactive feature comparison selector.

### 4. Valuation
Linear Regression modeling interface displaying the fitted regression equation:
$$\text{SalePrice} = 102.49 \times \text{GrLivArea} + \$24,899.75$$

Includes an interactive valuation calculator allowing users to enter living area in square feet and receive an instant valuation with input boundaries validation, clear assumptions, and disclaimers.

### 5. Model Performance
Generalization loss analysis evaluated on 292 unseen test listings (20% partition):
- **MAE:** \$38,341.20
- **MSE:** 3,418,946,311.18
- **RMSE:** \$58,471.76
- **Test $R^2$ Score:** 0.5543 (55.43% variance explained)
- Diagnostic Actual vs. Predicted scatter plot with identity reference line ($y = x$).
- Residual error distribution and discussion of single-variable model limitations.

---

## 4. Mathematical Model & Parameter Interpretation

### Ordinary Least Squares Formulation
$$\hat{Y} = m \cdot X + c$$

Where:
- $\hat{Y}$ is the predicted property sale price in USD.
- $X$ is the finished above-ground living area (`GrLivArea`) in square feet.
- $m = \$102.4890$ is the regression slope coefficient.
- $c = \$24,899.7482$ is the Y-intercept constant.

### Economic Interpretation:
- **Slope ($m = \$102.49$ / sq ft):** For every additional square foot of above-ground living area, property sale price increases by approximately \$102.49, assuming other property characteristics remain constant.
- **Intercept ($c = \$24,899.75$):** The equation baseline where living area equals zero. In residential real estate, a house of 0 square feet does not exist; therefore, the intercept acts as a mathematical calibration constant.

---

## 5. Installation & Execution

### Prerequisites
Python 3.10+ installed.

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Verification Test Suite
```bash
python test_app.py
```
*Expected: `ALL 8 ESTATEIQ VERIFICATION TESTS PASSED SUCCESSFULLY!`*

### 3. Start the Web Application
```bash
streamlit run app.py
```
Open your browser to: `http://localhost:8501`

### 4. Open the Jupyter Notebook
Open `notebooks/property_analysis.ipynb` in VS Code or JupyterLab to view the complete pre-executed step-by-step analysis with embedded visualisations.
