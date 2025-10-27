import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import pickle
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Set style for research paper
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16
})


# Load data and model
data_dir = Path("datasets")
models_dir = Path("models")
output_dir = Path("outputs")

# Load model and data
try:
    model = pickle.load(open(models_dir / "cashflow_model.pkl", "rb"))
    feature_map = pickle.load(open(models_dir / "feature_map.pkl", "rb"))
    print(" - Model loaded")
except:
    print(" - Model not found. Using sample data only.")
    model = None

train = pd.read_csv(data_dir / "train.csv", low_memory=False)
store = pd.read_csv(data_dir / "store.csv")
df = train.merge(store, on='Store', how='left')
df['Date'] = pd.to_datetime(df['Date'])
df = df[df['Open'] == 1].copy()


store_1 = df[df['Store'] == 1].sort_values('Date')
store_1 = store_1[store_1['Sales'] > 0].set_index('Date')['Sales']

if len(store_1) > 365:
    decomposition = seasonal_decompose(store_1, model='additive', period=365)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12))

    axes[0].plot(store_1.index, store_1.values, linewidth=2, color='black')
    axes[0].set_title('Original Revenue Time Series', fontweight='bold')
    axes[0].set_ylabel('Revenue ($)')
    axes[0].grid(True, alpha=0.3)

    # Trend component
    axes[1].plot(decomposition.trend.index, decomposition.trend.values, linewidth=2, color='red')
    axes[1].set_title('Trend Component (Mathematical Model)', fontweight='bold')
    axes[1].set_ylabel('Trend ($)')
    axes[1].grid(True, alpha=0.3)

    # Seasonal component
    axes[2].plot(decomposition.seasonal.index, decomposition.seasonal.values, linewidth=2, color='green')
    axes[2].set_title('Seasonal Component (Mathematical Model)', fontweight='bold')
    axes[2].set_ylabel('Seasonality ($)')
    axes[2].grid(True, alpha=0.3)

    # Residual component
    axes[3].plot(decomposition.resid.index, decomposition.resid.values, linewidth=2, color='blue')
    axes[3].set_title('Residual Component (Mathematical Model)', fontweight='bold')
    axes[3].set_ylabel('Residual ($)')
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'research_time_series_decomposition.png', dpi=300, bbox_inches='tight')
    plt.close()

if model is not None:
    df_model = df.copy()
    df_model['StateHoliday'] = df_model['StateHoliday'].astype(str).map({'0': 0, 'a': 1, 'b': 2, 'c': 3}).fillna(0).astype(int)
    
    # Time features (complete set)
    df_model['Year'] = df_model['Date'].dt.year
    df_model['Month'] = df_model['Date'].dt.month
    df_model['WeekOfYear'] = df_model['Date'].dt.isocalendar().week
    df_model['DayOfYear'] = df_model['Date'].dt.dayofyear
    df_model['IsMonthEnd'] = df_model['Date'].dt.is_month_end
    df_model['IsMonthStart'] = df_model['Date'].dt.is_month_start
    df_model['IsQuarterEnd'] = df_model['Date'].dt.is_quarter_end
    df_model['IsQuarterStart'] = df_model['Date'].dt.is_quarter_start
    
    # Cyclical encoding
    df_model['Month_sin'] = np.sin(2 * np.pi * df_model['Month'] / 12)
    df_model['Month_cos'] = np.cos(2 * np.pi * df_model['Month'] / 12)
    df_model['DayOfWeek_sin'] = np.sin(2 * np.pi * df_model['DayOfWeek'] / 7)
    df_model['DayOfWeek_cos'] = np.cos(2 * np.pi * df_model['DayOfWeek'] / 7)

    # Lag features (complete set)
    df_model = df_model.sort_values(['Store', 'Date'])
    for lag in [1, 2, 3, 7, 14, 21, 30]:
        df_model[f'sales_lag_{lag}'] = df_model.groupby('Store')['Sales'].shift(lag)

    # Rolling statistics (complete set)
    for window in [7, 14, 21, 30]:
        df_model[f'sales_rolling_mean_{window}'] = df_model.groupby('Store')['Sales'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).mean()
        )
        df_model[f'sales_rolling_std_{window}'] = df_model.groupby('Store')['Sales'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).std()
        )
        df_model[f'sales_rolling_max_{window}'] = df_model.groupby('Store')['Sales'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).max()
        )
        df_model[f'sales_rolling_min_{window}'] = df_model.groupby('Store')['Sales'].transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=1).min()
        )

    # Percent change features
    df_model['sales_pct_change'] = df_model.groupby('Store')['Sales'].pct_change()
    df_model['sales_pct_change_7d'] = df_model.groupby('Store')['Sales'].pct_change(7)

    # Interaction features
    df_model['promo_x_lag1'] = df_model['Promo'] * df_model['sales_lag_1']
    df_model['dayofweek_x_promo'] = df_model['DayOfWeek'] * df_model['Promo']
    df_model['holiday_x_promo'] = df_model['StateHoliday'] * df_model['Promo']

    # Customer features
    df_model['customers_lag_1'] = df_model.groupby('Store')['Customers'].shift(1)
    df_model['customers_lag_7'] = df_model.groupby('Store')['Customers'].shift(7)
    df_model['customers_rolling_mean_7'] = df_model.groupby('Store')['Customers'].transform(
        lambda x: x.shift(1).rolling(window=7, min_periods=1).mean()
    )

    df_model = pd.get_dummies(df_model, columns=['StoreType', 'Assortment'], drop_first=True)
    
    # Get expected feature columns from model
    feature_cols = [feature_map[i] for i in range(len(feature_map))]
    
    # Replace infinite values with NaN then fill like training
    df_model = df_model.replace([np.inf, -np.inf], np.nan)
    df_model = df_model.fillna(method='bfill').fillna(method='ffill').fillna(0)
    
    # Drop rows that don't have all required features
    df_model = df_model.dropna(subset=feature_cols + ['Sales'])

    # Time series split
    split_date = pd.to_datetime('2015-07-01')
    val_data = df_model[df_model['Date'] >= split_date].copy()

    # feature_cols already defined above
    # Now use only existing columns
    existing_cols = [col for col in feature_cols if col in val_data.columns]

    X_val = val_data[existing_cols]
    y_val = val_data['Sales']
    y_pred = model.predict(X_val)

    # Create comprehensive accuracy analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Actual vs Predicted with perfect prediction line
    axes[0, 0].scatter(y_val, y_pred, alpha=0.5, s=20, color='steelblue')
    min_val, max_val = min(y_val.min(), y_pred.min()), max(y_val.max(), y_pred.max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual Revenue ($)', fontsize=12)
    axes[0, 0].set_ylabel('Predicted Revenue ($)', fontsize=12)
    axes[0, 0].set_title('Model Accuracy: Actual vs Predicted Revenue\n(Multiple Linear Regression Visualization)', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Add correlation coefficient
    correlation = np.corrcoef(y_val, y_pred)[0, 1]
    axes[0, 0].text(0.05, 0.95, f'Correlation: {correlation:.3f}', transform=axes[0, 0].transAxes,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Error distribution (should be normal for good model)
    errors = y_val - y_pred
    axes[0, 1].hist(errors, bins=50, color='coral', alpha=0.7, edgecolor='black', density=True)

    # Add normal distribution curve
    mu, sigma = errors.mean(), errors.std()
    x = np.linspace(errors.min(), errors.max(), 100)
    axes[0, 1].plot(x, stats.norm.pdf(x, mu, sigma), 'r-', linewidth=2, label='Normal Distribution')

    axes[0, 1].axvline(mu, color='red', linestyle='--', label=f'Mean: ${mu:,.0f}')
    axes[0, 1].set_xlabel('Prediction Error ($)', fontsize=12)
    axes[0, 1].set_ylabel('Density', fontsize=12)
    axes[0, 1].set_title('Error Distribution Analysis\n(Statistical Model Validation)', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Performance over time
    val_data = val_data.assign(predicted=y_pred, error=errors)
    daily_performance = val_data.groupby('Date').agg({
        'Sales': 'mean',
        'predicted': 'mean',
        'error': 'mean'
    }).rolling(7).mean()

    axes[1, 0].plot(daily_performance.index, daily_performance['Sales'], 'b-', linewidth=2, label='Actual')
    axes[1, 0].plot(daily_performance.index, daily_performance['predicted'], 'r--', linewidth=2, label='Predicted')
    axes[1, 0].fill_between(daily_performance.index,
                          daily_performance['Sales'] - daily_performance['error'].abs(),
                          daily_performance['Sales'] + daily_performance['error'].abs(),
                          alpha=0.3, color='gray', label='Error Range')
    axes[1, 0].set_xlabel('Date', fontsize=12)
    axes[1, 0].set_ylabel('Revenue ($)', fontsize=12)
    axes[1, 0].set_title('Model Performance Over Time\n(7-day Rolling Average)', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)

    # Feature importance (mathematical model interpretation)
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False).head(10)

    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(importance_df)))
    axes[1, 1].barh(range(len(importance_df)), importance_df['importance'], color=colors)
    axes[1, 1].set_yticks(range(len(importance_df)))
    axes[1, 1].set_yticklabels(importance_df['feature'], fontsize=10)
    axes[1, 1].set_xlabel('Feature Importance (Gini Importance)', fontsize=12)
    axes[1, 1].set_title('Feature Importance Analysis\n(Mathematical Model Interpretation)', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='x')

    # Add cumulative importance
    cumulative_importance = importance_df['importance'].cumsum()
    axes[1, 1].axvline(cumulative_importance.iloc[4], color='red', linestyle='--', alpha=0.7,
                      label=f'80% importance\n(top 5 features)')

    plt.tight_layout()
    plt.savefig(output_dir / 'research_model_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()

# === VISUALIZATION 3: Business Insights (Statistical Analysis) ===
print("\n3. Creating business insights analysis...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Weekly patterns with confidence intervals
weekly_stats = df.groupby('DayOfWeek')['Sales'].agg(['mean', 'std', 'count']).reset_index()
day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

# Calculate confidence intervals
weekly_stats['ci_lower'] = weekly_stats['mean'] - 1.96 * weekly_stats['std'] / np.sqrt(weekly_stats['count'])
weekly_stats['ci_upper'] = weekly_stats['mean'] + 1.96 * weekly_stats['std'] / np.sqrt(weekly_stats['count'])

axes[0, 0].bar(range(1, 8), weekly_stats['mean'], yerr=weekly_stats['std'], capsize=5, color='steelblue', alpha=0.7)
axes[0, 0].set_xticks(range(1, 8))
axes[0, 0].set_xticklabels(day_names)
axes[0, 0].set_xlabel('Day of Week', fontsize=12)
axes[0, 0].set_ylabel('Average Revenue ($)', fontsize=12)
axes[0, 0].set_title('Weekly Revenue Patterns\n(Statistical Analysis with Confidence Intervals)', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Store type performance comparison
store_type_performance = df.groupby('StoreType')['Sales'].agg(['mean', 'std', 'count'])
store_type_performance['ci_lower'] = store_type_performance['mean'] - 1.96 * store_type_performance['std'] / np.sqrt(store_type_performance['count'])
store_type_performance['ci_upper'] = store_type_performance['mean'] + 1.96 * store_type_performance['std'] / np.sqrt(store_type_performance['count'])

x_pos = range(len(store_type_performance))
axes[0, 1].bar(x_pos, store_type_performance['mean'], yerr=store_type_performance['std'], capsize=5, color='coral', alpha=0.7)
axes[0, 1].set_xticks(x_pos)
axes[0, 1].set_xticklabels(store_type_performance.index)
axes[0, 1].set_xlabel('Store Type', fontsize=12)
axes[0, 1].set_ylabel('Average Revenue ($)', fontsize=12)
axes[0, 1].set_title('Store Type Performance Analysis\n(Statistical Comparison)', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Promotion effect analysis
promo_effect = df.groupby('Promo')['Sales'].agg(['mean', 'std', 'count'])
promo_effect['ci_lower'] = promo_effect['mean'] - 1.96 * promo_effect['std'] / np.sqrt(promo_effect['count'])
promo_effect['ci_upper'] = promo_effect['mean'] + 1.96 * promo_effect['std'] / np.sqrt(promo_effect['count'])

promo_lift = ((promo_effect.loc[1, 'mean'] - promo_effect.loc[0, 'mean']) / promo_effect.loc[0, 'mean']) * 100

axes[1, 0].bar(['No Promo', 'Promo'], promo_effect['mean'], yerr=promo_effect['std'], capsize=5, color=['lightcoral', 'lightgreen'], alpha=0.7)
axes[1, 0].set_xlabel('Promotion Status', fontsize=12)
axes[1, 0].set_ylabel('Average Revenue ($)', fontsize=12)
axes[1, 0].set_title(f'Promotion Effect Analysis\n(Statistical Significance: {promo_lift:.1f}% Average Lift)', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Revenue distribution by store type
store_types = df['StoreType'].unique()
colors = ['blue', 'green', 'red', 'orange']

for i, store_type in enumerate(store_types):
    type_data = df[df['StoreType'] == store_type]['Sales']
    axes[1, 1].hist(type_data, bins=30, alpha=0.6, label=f'Type {store_type}', color=colors[i])

axes[1, 1].set_xlabel('Revenue ($)', fontsize=12)
axes[1, 1].set_ylabel('Frequency', fontsize=12)
axes[1, 1].set_title('Revenue Distribution by Store Type\n(Statistical Distribution Analysis)', fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'research_business_insights.png', dpi=300, bbox_inches='tight')
plt.close()

# === VISUALIZATION 4: Mathematical Model Validation ===
print("\n4. Creating mathematical model validation...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Autocorrelation analysis (shows if time series is predictable)
store_sample = df[df['Store'] == 1].sort_values('Date')
if len(store_sample) > 100:
    sales_ts = store_sample.set_index('Date')['Sales'].ffill()

    # ACF plot
    plot_acf(sales_ts, ax=axes[0, 0], lags=30, alpha=0.05)
    axes[0, 0].set_title('Autocorrelation Function (ACF)\n(Mathematical Model: AR Process)', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # PACF plot
    plot_pacf(sales_ts, ax=axes[0, 1], lags=30, alpha=0.05)
    axes[0, 1].set_title('Partial Autocorrelation Function (PACF)\n(Mathematical Model: MA Process)', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

# Error analysis by prediction horizon (if we have multi-step predictions)
if model is not None and len(val_data) > 0:
    # Calculate performance metrics
    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)

    # Error by store type
    val_data_with_pred = val_data.assign(predicted=y_pred)
    # Merge back store information
    val_data_with_pred = val_data_with_pred.merge(store[['Store', 'StoreType']], on='Store', how='left')
    error_by_type = val_data_with_pred.groupby('StoreType').apply(
        lambda x: mean_absolute_error(x['Sales'], x['predicted'])
    )

    axes[1, 0].bar(range(len(error_by_type)), error_by_type.values, color='purple', alpha=0.7)
    axes[1, 0].set_xticks(range(len(error_by_type)))
    axes[1, 0].set_xticklabels(error_by_type.index)
    axes[1, 0].set_xlabel('Store Type', fontsize=12)
    axes[1, 0].set_ylabel('Mean Absolute Error ($)', fontsize=12)
    axes[1, 0].set_title('Model Error by Store Type\n(Statistical Performance Analysis)', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')

    # Add performance metrics
    axes[1, 0].text(0.02, 0.98, f'Overall MAE: ${mae:,.0f}\nOverall R²: {r2:.3f}',
                    transform=axes[1, 0].transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Q-Q plot for error normality
if model is not None and len(val_data) > 0:
    errors = y_val - y_pred
    stats.probplot(errors, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot: Error Normality\n(Statistical Assumption Validation)', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / 'research_model_validation.png', dpi=300, bbox_inches='tight')
plt.close()

