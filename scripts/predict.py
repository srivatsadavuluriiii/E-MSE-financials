import pandas as pd
import numpy as np
import pickle
from pathlib import Path

def predict_cashflow(store_id, forecast_days=30):
    # Load model and features
    models_dir = Path("models")
    
    try:
        model = pickle.load(open(models_dir / "cashflow_model.pkl", "rb"))
        feature_map = pickle.load(open(models_dir / "feature_map.pkl", "rb"))
    except FileNotFoundError:
        print("Error: Model not found. Run train.py first!")
        return None
    
    # Load data
    data_dir = Path("datasets")
    train = pd.read_csv(data_dir / "train.csv")
    store = pd.read_csv(data_dir / "store.csv")
    
    # Get historical data for this store
    df = train[train['Store'] == store_id].copy()
    df = df.merge(store[store['Store'] == store_id], on='Store', how='left')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df[df['Open'] == 1].sort_values('Date')
    
    if len(df) == 0:
        print(f"No data found for Store {store_id}")
        return None
    
    print(f"\nPredicting cash flow for Store {store_id}")
    print(f"Historical period: {df['Date'].min()} to {df['Date'].max()}")
    
    # Create same features as training
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['WeekOfYear'] = df['Date'].dt.isocalendar().week
    df['IsMonthEnd'] = df['Date'].dt.is_month_end
    df['IsMonthStart'] = df['Date'].dt.is_month_start
    
    for lag in [1, 7, 14, 30]:
        df[f'sales_lag_{lag}'] = df['Sales'].shift(lag)
        
    for window in [7, 14, 30]:
        df[f'sales_rolling_mean_{window}'] = df['Sales'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
        df[f'sales_rolling_std_{window}'] = df['Sales'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
        )
    
    df['customers_lag_7'] = df['Customers'].shift(7)
    
    # One-hot encode store features
    df = pd.get_dummies(df, columns=['StoreType', 'Assortment'], drop_first=True)
    
    # Get the most recent data point
    latest = df.iloc[-1].copy()
    
    # Fix StateHoliday encoding
    if isinstance(latest['StateHoliday'], str):
        latest['StateHoliday'] = {'0': 0, 'a': 1, 'b': 2, 'c': 3}.get(latest['StateHoliday'], 0)
    
    # Create a simple forecast using recent trend
    recent_avg = df['Sales'].tail(forecast_days).mean()
    recent_std = df['Sales'].tail(forecast_days).std()
    
    # Predict - only use features that exist
    feature_cols = [feature_map[i] for i in range(len(feature_map))]
    existing_cols = [col for col in feature_cols if col in latest.index]
    
    # Create array with 0s for missing columns
    X_values = [latest.get(col, 0) for col in feature_cols]
    X_pred = np.array([X_values]).reshape(1, -1)
    
    prediction = model.predict(X_pred)[0]
    
    print(f" - REVENUE FORECAST SUMMARY (Store {store_id})")
    print(f"\n - Recent Average Daily Revenue: ${recent_avg:,.2f}")
    print(f"\n - Predicted Next Day Revenue: ${prediction:,.2f}")

    # Project forward with confidence intervals
    lower = prediction - (recent_std * 1.96)  # 95% CI
    upper = prediction + (recent_std * 1.96)

    print(f"\n30-Day Revenue Forecast:")
    print(f"  Expected: ${prediction * 30:,.2f}")
    print(f"  Range: ${lower * 30:,.2f} - ${upper * 30:,.2f}")

    print(f"\n60-Day Revenue Forecast:")
    print(f"  Expected: ${prediction * 60:,.2f}")
    print(f"  Range: ${lower * 60:,.2f} - ${upper * 60:,.2f}")

    print(f"\n90-Day Revenue Forecast:")
    print(f"  Expected: ${prediction * 90:,.2f}")
    print(f"  Range: ${lower * 90:,.2f} - ${upper * 90:,.2f}")

    print(f"\nNote: This forecasts revenue, not cash flow.")
    print(f"Revenue forecasting is a key component of cash flow management.")
    
    
    return {
        'store_id': store_id,
        'predicted_daily': prediction,
        'forecast_30d': prediction * 30,
        'forecast_60d': prediction * 60,
        'forecast_90d': prediction * 90,
        'confidence_lower': lower,
        'confidence_upper': upper
    }

if __name__ == "__main__":
    # example: Predict for Store 1
    result = predict_cashflow(store_id=1, forecast_days=30)

