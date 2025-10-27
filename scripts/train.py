
import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle

# Load data
data_dir = Path("datasets")
train = pd.read_csv(data_dir / "train.csv", low_memory=False)
store = pd.read_csv(data_dir / "store.csv", low_memory=False)

df = train.merge(store, on='Store', how='left')
df['Date'] = pd.to_datetime(df['Date'])
df = df[df['Open'] == 1].copy()

# 2. Enhanced Feature Engineering

# Convert StateHoliday
df['StateHoliday'] = df['StateHoliday'].astype(str)
df['StateHoliday'] = df['StateHoliday'].map({'0': 0, 'a': 1, 'b': 2, 'c': 3}).fillna(0).astype(int)

# Time features
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['WeekOfYear'] = df['Date'].dt.isocalendar().week
df['DayOfYear'] = df['Date'].dt.dayofyear
df['IsMonthEnd'] = df['Date'].dt.is_month_end
df['IsMonthStart'] = df['Date'].dt.is_month_start
df['IsQuarterEnd'] = df['Date'].dt.is_quarter_end
df['IsQuarterStart'] = df['Date'].dt.is_quarter_start

# Cyclical encoding for time features
df['Month_sin'] = np.sin(2 * np.pi * df['Month'] / 12)
df['Month_cos'] = np.cos(2 * np.pi * df['Month'] / 12)
df['DayOfWeek_sin'] = np.sin(2 * np.pi * df['DayOfWeek'] / 7)
df['DayOfWeek_cos'] = np.cos(2 * np.pi * df['DayOfWeek'] / 7)

# Lag features (extended)
df = df.sort_values(['Store', 'Date'])
for lag in [1, 2, 3, 7, 14, 21, 30]:
    df[f'sales_lag_{lag}'] = df.groupby('Store')['Sales'].shift(lag)

# Rolling statistics (extended windows)
for window in [7, 14, 21, 30]:
    df[f'sales_rolling_mean_{window}'] = df.groupby('Store')['Sales'].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
    )
    df[f'sales_rolling_std_{window}'] = df.groupby('Store')['Sales'].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
    )
    df[f'sales_rolling_max_{window}'] = df.groupby('Store')['Sales'].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).max()
    )
    df[f'sales_rolling_min_{window}'] = df.groupby('Store')['Sales'].transform(
        lambda x: x.shift(1).rolling(window=window, min_periods=1).min()
    )

# Percent change features
df['sales_pct_change'] = df.groupby('Store')['Sales'].pct_change()
df['sales_pct_change_7d'] = df.groupby('Store')['Sales'].pct_change(7)

# Interaction features
df['promo_x_lag1'] = df['Promo'] * df['sales_lag_1']
df['dayofweek_x_promo'] = df['DayOfWeek'] * df['Promo']
df['holiday_x_promo'] = df['StateHoliday'] * df['Promo']

# Customer features
df['customers_lag_1'] = df.groupby('Store')['Customers'].shift(1)
df['customers_lag_7'] = df.groupby('Store')['Customers'].shift(7)
df['customers_rolling_mean_7'] = df.groupby('Store')['Customers'].transform(
    lambda x: x.shift(1).rolling(window=7, min_periods=1).mean()
)

# Store features
df = pd.get_dummies(df, columns=['StoreType', 'Assortment'], drop_first=True)


# 3. Select features for training
feature_cols = [
    'DayOfWeek', 'Promo', 'StateHoliday', 'SchoolHoliday',
    'Year', 'Month', 'WeekOfYear', 'DayOfYear',
    'IsMonthEnd', 'IsMonthStart', 'IsQuarterEnd', 'IsQuarterStart',
    'Month_sin', 'Month_cos', 'DayOfWeek_sin', 'DayOfWeek_cos',
    'CompetitionDistance', 'Promo2',
    'sales_lag_1', 'sales_lag_2', 'sales_lag_3', 'sales_lag_7', 
    'sales_lag_14', 'sales_lag_21', 'sales_lag_30',
    'sales_rolling_mean_7', 'sales_rolling_std_7', 
    'sales_rolling_max_7', 'sales_rolling_min_7',
    'sales_rolling_mean_14', 'sales_rolling_std_14',
    'sales_rolling_mean_30', 'sales_rolling_std_30',
    'sales_pct_change', 'sales_pct_change_7d',
    'promo_x_lag1', 'dayofweek_x_promo', 'holiday_x_promo',
    'customers_lag_1', 'customers_lag_7', 'customers_rolling_mean_7'
] + [col for col in df.columns if col.startswith('StoreType_') or col.startswith('Assortment_')]

# Remove any columns that don't exist
feature_cols = [col for col in feature_cols if col in df.columns]

# Drop rows with NaN
df = df.dropna(subset=feature_cols + ['Sales'])

# Replace infinite values with NaN then fill
df = df.replace([np.inf, -np.inf], np.nan)
df = df.fillna(method='bfill').fillna(method='ffill').fillna(0)

print(f"\n - SELECTED {len(feature_cols)} FEATURES (up from baseline)")

# 4. Time-series split
split_date = pd.to_datetime('2015-07-01')

train_data = df[df['Date'] < split_date].copy()
val_data = df[df['Date'] >= split_date].copy()

print(f"   Train: {len(train_data)} samples")
print(f"   Validation: {len(val_data)} samples")

X_train = train_data[feature_cols]
y_train = train_data['Sales']

X_val = val_data[feature_cols]
y_val = val_data['Sales']

# 5. Train IMPROVED XGBoost model

model = XGBRegressor(
    n_estimators=200,           # More trees
    max_depth=8,                # Deeper trees
    learning_rate=0.05,         # Lower learning rate
    subsample=0.8,              # Row sampling
    colsample_bytree=0.8,       # Column sampling
    min_child_weight=3,         # Regularization
    gamma=0.1,                  # Regularization
    reg_alpha=0.1,              # L1 regularization
    reg_lambda=1.0,              # L2 regularization
    random_state=42,
    tree_method='hist'
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)


# 6. Evaluate

y_pred = model.predict(X_val)

mae = mean_absolute_error(y_val, y_pred)
rmse = np.sqrt(mean_squared_error(y_val, y_pred))
r2 = r2_score(y_val, y_pred)

direction_actual = np.diff(y_val.values) > 0
direction_pred = np.diff(y_pred) > 0
direction_acc = np.mean(direction_actual == direction_pred) * 100

print(f"   MAE: ${mae:,.2f}")
print(f"   RMSE: ${rmse:,.2f}")
print(f"   R²: {r2:.4f}")
print(f"   Directional Accuracy: {direction_acc:.1f}%")

avg_sales = y_val.mean()
mae_pct = (mae / avg_sales) * 100
print(f"   MAE as % of Average: {mae_pct:.2f}%")

# Compare to baseline
improvement_r2 = (r2 - 0.90) * 100
improvement_mae_pct = (9.6 - mae_pct)
print(f"   R² improvement: {improvement_r2:+.1f} points")
print(f"   MAE % improvement: {improvement_mae_pct:+.2f} percentage points")

output_dir = Path("models")
output_dir.mkdir(exist_ok=True)

pickle.dump(model, open(output_dir / "cashflow_model.pkl", "wb"))
feature_map = {i: feature_cols[i] for i in range(len(feature_cols))}
pickle.dump(feature_map, open(output_dir / "feature_map.pkl", "wb"))


# Feature importance -> top 15 important features
importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print(importance.head(15).to_string(index=False))


print(f"  - Can predict daily revenue with ${mae:,.0f} average error")
print(f"  - Accuracy: {mae_pct:.1f}% of average daily revenue")
print(f"  - Variance explained: {r2:.1%}")
print(f"  - Directional prediction: {direction_acc:.0f}% correct")


