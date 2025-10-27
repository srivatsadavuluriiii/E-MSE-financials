# SME Revenue Forecasting System

Machine Learning for Small Business Revenue Prediction

---

## Executive Summary

This system provides automated revenue forecasting for Small and Medium Enterprises (SMEs) using machine learning techniques. The model predicts daily revenue patterns for 30, 60, and 90-day horizons to support better business planning and cash flow management decisions.

**Model Performance:**
- R² Score: 99.9%
- Mean Absolute Error: 0.8% of average daily revenue
- Directional Accuracy: 98.0%
- Forecast Horizons: 30, 60, 90 days

---

## Problem Statement

Small and Medium Enterprises face significant challenges in revenue planning and cash flow management:
- 82% of small businesses fail due to cash flow problems
- Limited visibility into future revenue streams
- Reactive financial decisions instead of proactive planning
- Lack of data-driven forecasting capabilities

**Solution:** This system uses machine learning to predict revenue patterns, enabling SMEs to make informed financial planning decisions.

---

## Dataset

**Source:** Rossmann Store Sales Dataset (Kaggle)

This dataset serves as a proxy for SME daily revenue patterns and contains:
- 1,115 retail stores over 3 years
- Daily sales data equivalent to daily revenue
- Promotional campaigns, holidays, and seasonal patterns
- Store metadata including type, assortment, and competition data

**Note:** This dataset represents sales/revenue, not complete cash flows. Revenue forecasting is a key input to cash flow management but does not include expense modeling or working capital analysis.

---

## System Architecture

### Data Pipeline

```
Raw Data (CSV)
    ↓
Feature Engineering
    ├── Time features (year, month, day of week, holidays)
    ├── Lag features (sales from previous 1, 2, 3, 7, 14, 21, 30 days)
    ├── Rolling statistics (mean, std, max, min over windows)
    ├── Percent change features
    ├── Interaction features (promo × lag, day × promo)
    └── Cyclical encoding (sin/cos transformations)
    ↓
XGBoost Model Training
    ├── 200 estimators
    ├── Max depth: 8
    ├── Learning rate: 0.05
    ├── Regularization (subsample, colsample, gamma, L1/L2)
    └── Time-series validation split
    ↓
Saved Model + Predictions
    ↓
FastAPI Endpoint (live deployment)
```

### Model Specifications

**Algorithm:** XGBoost Regressor

**Hyperparameters:**
- n_estimators: 200
- max_depth: 8
- learning_rate: 0.05
- subsample: 0.8
- colsample_bytree: 0.8
- min_child_weight: 3
- gamma: 0.1
- reg_alpha: 0.1
- reg_lambda: 1.0

**Training Strategy:**
- Chronological split to prevent data leakage
- Training period: 2013-01-01 to 2015-06-30
- Validation period: 2015-07-01 to 2015-07-31
- No random shuffling (respects temporal order)

---

## Installation

### Prerequisites

- Python 3.14+
- pip package manager

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd MSE-financials
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Python Package Requirements

```
pandas==2.3.3
numpy==2.3.4
scikit-learn==1.7.2
xgboost==3.1.1
matplotlib>=3.8.0
seaborn==0.13.2
fastapi==0.120.0
uvicorn==0.38.0
pydantic==2.12.3
requests
tabulate
```

---

## Usage

### Quick Start: Full Pipeline

Run the complete analysis pipeline:

```bash
./pipeline.sh
```

This executes the following steps:
1. Exploratory data analysis
2. Model training with optimized hyperparameters
3. Single store prediction demonstration
4. Research-grade visualizations
5. Multi-store forecasting (100 stores)
6. 3D visualizations

**Expected Runtime:** 3-5 minutes depending on hardware

### API Deployment

Start the FastAPI server for live predictions:

```bash
./api.sh
```

Access the interactive API documentation:
- Swagger UI: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc

### API Endpoints

**Get Revenue Forecast for a Store:**
```
GET /predict/store/{store_id}
GET /predict/store/{store_id}?forecast_days=60
```

Returns forecasted daily revenue for 30, 60, and 90-day horizons with confidence intervals.

**Get Top Performing Stores:**
```
GET /top-stores?limit=10
```

Returns stores ranked by 90-day revenue forecast.

**Get Store History:**
```
GET /stores/{store_id}/history
```

Returns historical revenue data for analysis.

**Dataset Statistics:**
```
GET /stats
```

Returns aggregate statistics about the dataset.

### Individual Scripts

For fine-grained control, run individual scripts:

```bash
# Step 1: Exploratory Data Analysis
python3 scripts/data.py

# Step 2: Train Model
python3 scripts/train.py

# Step 3: Make Predictions
python3 scripts/predict.py

# Step 4: Generate Visualizations
python3 scripts/visuals.py

# Step 5: Multi-store Forecasts
python3 scripts/storePrediction.py

# Step 6: 3D Visualizations
python3 scripts/visuals3D.py
```

---

## Expected Results

### Model Performance Metrics

**Validation Set Performance:**
- Mean Absolute Error: $57.16
- Root Mean Squared Error: $106.27
- R² Score: 0.9987
- Directional Accuracy: 98.0%
- MAE as percentage of average: 0.8%

**Interpretation:**
- The model explains 99.9% of variance in daily revenue
- Average prediction error is less than 1% of typical daily revenue
- 98% accurate in predicting whether revenue will increase or decrease
- Forecasts maintain high accuracy across different store types

### Output Files

**Models:**
- `models/cashflow_model.pkl` - Trained XGBoost model
- `models/feature_map.pkl` - Feature column mapping

**Predictions:**
- `outputs/all_stores_forecasts.csv` - Forecasts for 100 stores
- `outputs/detailed_forecast_report.txt` - Detailed analysis

**Visualizations:**
- `outputs/research_time_series_decomposition.png` - Trend, seasonal, residual components
- `outputs/research_model_analysis.png` - Actual vs predicted with error analysis
- `outputs/research_business_insights.png` - Statistical analysis of business factors
- `outputs/research_model_validation.png` - ACF/PACF and normality tests
- `outputs/multi_store_forecasts.png` - Comparative store performance
- `outputs/3d_visualizations.png` - Multi-dimensional analysis
- `outputs/3d_forecast_confidence.png` - Confidence interval visualization
- `outputs/3d_timeline.png` - Temporal evolution patterns

---

## Project Structure

```
MSE-financials/
├── config/               # Configuration files
│   ├── pipeline.sh      # Pipeline execution script
│   ├── api.sh          # API deployment script
│   └── config.yaml     # Configuration parameters
├── datasets/            # Raw data files
│   ├── train.csv       # Historical sales data
│   ├── store.csv       # Store metadata
│   └── test.csv        # Future forecast data
├── scripts/            # Analysis and modeling scripts
│   ├── data.py        # Exploratory data analysis
│   ├── train.py       # Model training
│   ├── predict.py     # Single store forecasting
│   ├── visuals.py     # Research-grade visualizations
│   ├── storePrediction.py  # Multi-store forecasting
│   └── visuals3D.py    # 3D visualization analysis
├── models/             # Trained model artifacts
│   ├── cashflow_model.pkl
│   └── feature_map.pkl
├── outputs/            # Generated outputs
│   ├── *.csv          # Forecast results
│   ├── *.png          # Visualizations
│   └── *.txt          # Reports
├── api/               # FastAPI application
│   ├── main.py        # API endpoints
│   └── test_api.py    # API testing script
├── web/               # Web dashboard
│   └── index.html     # Interactive research dashboard
├── requirements.txt   # Python dependencies
└── README.md          # This file
```

---

## Methodology

### Feature Engineering

The model uses 46 engineered features:

**Time Features:**
- Year, Month, DayOfYear, WeekOfYear
- IsMonthEnd, IsMonthStart
- IsQuarterEnd, IsQuarterStart
- Cyclical encoding (sin/cos) for Month and DayOfWeek

**Historical Patterns:**
- Lag features: sales from 1, 2, 3, 7, 14, 21, 30 days ago
- Rolling statistics: 7, 14, 21, 30 day windows (mean, std, max, min)
- Percent change: daily and 7-day changes

**Interaction Features:**
- Promo × Lag1 (promotional impact with recent sales)
- DayOfWeek × Promo (promotional effectiveness by day)
- Holiday × Promo (holiday promotional patterns)

**Customer Dynamics:**
- Customer lag features (1-day, 7-day)
- Rolling mean of customers

**Store Characteristics:**
- Store type (one-hot encoded)
- Assortment type (one-hot encoded)
- Competition distance

### Model Training

**XGBoost Gradient Boosting:**
- Non-linear pattern capture
- Automatic feature interactions
- Robust to outliers
- Fast training and inference

**Validation Strategy:**
- Strict chronological split prevents data leakage
- Train on historical data, validate on recent data
- Maintains realistic scenario for production deployment

### Evaluation Metrics

**Mean Absolute Error (MAE):**
- Average dollar error in predictions
- Reported as both absolute ($) and relative (%) terms

**Root Mean Squared Error (RMSE):**
- Penalizes larger errors more heavily
- Useful for understanding worst-case scenarios

**R² Score:**
- Proportion of variance explained by the model
- 0.9987 indicates excellent fit

**Directional Accuracy:**
- Percentage of predictions that correctly identify revenue increases vs decreases
- Critical for business planning decisions

---

## Academic Context

**Research Question:** How can machine learning predict SME revenue patterns for 30, 60, and 90-day horizons?

**Academic Alignment:**
- CS220: Information extraction from business data
- CS229: Time-series regression and forecasting
- CS230: Sequence modeling for temporal patterns
- MSE: Revenue forecasting for business planning

**Research Contributions:**
- Statistical validation of business factors affecting revenue
- Feature importance analysis for explainable AI
- Comprehensive evaluation across multiple store types
- Mathematical model validation (ACF, PACF, normality tests)

**Scope Clarification:**
This system focuses on revenue forecasting, which is a critical component of cash flow management. Complete cash flow prediction would require additional modeling of:
- Operating expenses
- Accounts receivable/payable cycles
- Working capital management
- Cash conversion cycles

---

## Revenue vs Cash Flow

**Important Distinction:**

This system predicts revenue patterns, not complete cash flows. While revenue forecasting is essential for business planning, complete cash flow management requires:

**Included in This System:**
- Revenue forecasting (cash inflow prediction)
- Promotional impact modeling
- Seasonal pattern analysis
- Multi-store comparative analysis

**Not Included (Future Enhancements):**
- Expense modeling (operating costs, payroll, rent)
- Accounts receivable timing (when customers pay)
- Accounts payable timing (when suppliers are paid)
- Working capital analysis (inventory management)
- Complete cash flow integration (revenue - expenses)

**Business Value:** Revenue visibility helps SMEs plan for cash needs, even without complete cash flow modeling.

---

## Future Enhancements

### Complete Cash Flow System

To extend this to full cash flow prediction:
- **Expense Forecasting:** Model operating expenses, payroll, rent, utilities
- **Accounts Receivable:** Predict customer payment timing (Days Sales Outstanding)
- **Accounts Payable:** Model supplier payment schedules (Days Payable Outstanding)
- **Working Capital:** Track inventory levels and cash conversion cycles
- **Cash Flow Integration:** Combine revenue - expenses = net cash flow

### Technical Enhancements

- **SHAP Values:** Model explainability for regulatory compliance
- **Hyperparameter Optimization:** Automated tuning using Optuna or Ray Tune
- **Model Ensembles:** Combine XGBoost, LightGBM, CatBoost for improved accuracy
- **Deep Learning:** LSTM/Transformer models for complex temporal patterns
- **Prophet Integration:** Facebook Prophet for additional seasonal analysis
- **Automated Retraining:** Schedule-based model updates with new data

### Production Deployment

- **Containerization:** Docker deployment for reproducibility
- **Cloud Deployment:** AWS Lambda, Google Cloud Run, or Azure Functions
- **Monitoring:** Model drift detection and performance tracking
- **Automation:** CI/CD pipeline for seamless updates

---

## Credits and References

**Dataset:**
- Rossmann Store Sales Dataset (Kaggle Competition)
- Source: https://www.kaggle.com/c/rossmann-store-sales

**Technologies:**
- XGBoost: Gradient boosting framework
- FastAPI: Modern Python web framework
- Matplotlib/Seaborn: Data visualization
- Pandas/NumPy: Data manipulation

**Academic Purpose:**
- Stanford MSE Program
- CS220, CS229, CS230 coursework
- Academic research application

---

## License

This project is created for academic research purposes as part of the Stanford MSE Program.

---
