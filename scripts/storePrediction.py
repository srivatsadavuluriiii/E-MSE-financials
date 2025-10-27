import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


# Load data
data_dir = Path("datasets")
train = pd.read_csv(data_dir / "train.csv", low_memory=False)
store = pd.read_csv(data_dir / "store.csv")

# Load model
models_dir = Path("models")
try:
    model = pickle.load(open(models_dir / "cashflow_model.pkl", "rb"))
    feature_map = pickle.load(open(models_dir / "feature_map.pkl", "rb"))
    print(" - Model loaded successfully")
except FileNotFoundError:
    print(" - Error: Model not found. Run train.py first!")
    exit(1)

# Feature columns
feature_cols = [feature_map[i] for i in range(len(feature_map))]


all_stores = train['Store'].unique()
print(f"Found {len(all_stores)} stores in dataset")
print(f" - Predicting for first 100 stores")


results = []
processed = 0

for store_id in all_stores[:100]:  # First 100 stores
    try:
        # Get store data
        df = train[train['Store'] == store_id].copy()
        df_store = store[store['Store'] == store_id].iloc[0]
        
        df['Date'] = pd.to_datetime(df['Date'])
        df = df[df['Open'] == 1].sort_values('Date')
        
        if len(df) < 100:  # Need at least 100 days
            continue
            
        # Simple feature extraction - just get averages and recent trends
        recent_avg = df['Sales'].tail(30).mean()
        recent_median = df['Sales'].tail(30).median()
        total_avg = df['Sales'].mean()
        
        # Use recent median as proxy prediction (simple baseline)
        prediction = recent_median
        
        # Calculate projections
        forecast_30d = prediction * 30
        forecast_60d = prediction * 60  
        forecast_90d = prediction * 90
        
        results.append({
            'Store': store_id,
            'StoreType': df_store['StoreType'],
            'Assortment': df_store['Assortment'],
            'Recent_Avg_Daily': recent_avg,
            'Predicted_Daily': prediction,
            'Historic_Avg': total_avg,
            'Forecast_30d': forecast_30d,
            'Forecast_60d': forecast_60d,
            'Forecast_90d': forecast_90d,
            'Data_Points': len(df)
        })
        
        processed += 1
        if processed % 10 == 0:
            print(f" - Processed {processed} stores")
            
    except Exception as e:
        continue

# Convert to DataFrame
results_df = pd.DataFrame(results)

# Save results
output_dir = Path("outputs")
results_df.to_csv(output_dir / 'all_stores_forecasts.csv', index=False)
print(f"\n Saved: outputs/all_stores_forecasts.csv ({len(results_df)} stores)")

# Print summary statistics

print(f"\nTotal stores analyzed: {len(results_df)}")
print(f"\nRevenue Forecast Statistics:")
print(f"  - Average daily revenue: ${results_df['Predicted_Daily'].mean():,.2f}")
print(f"  - Average 30-day revenue: ${results_df['Forecast_30d'].mean():,.0f}")
print(f"  - Average 60-day revenue: ${results_df['Forecast_60d'].mean():,.0f}")
print(f"  - Average 90-day revenue: ${results_df['Forecast_90d'].mean():,.0f}")

# Top 10 stores by forecast
top_stores = results_df.nlargest(10, 'Forecast_90d')[['Store', 'StoreType', 'Forecast_90d']]
print("\nTop 10 Stores by 90-Day Revenue Forecast:")
for idx, row in top_stores.iterrows():
    print(f"  Store {row['Store']:3d} ({row['StoreType']}): ${row['Forecast_90d']:>10,.0f}")

# Create visualization
print("\nGenerating visualization...")

from matplotlib import pyplot as plt
import seaborn as sns

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 1. 30-day forecasts by store
top_20_30d = results_df.nlargest(20, 'Forecast_30d')
axes[0, 0].barh(range(len(top_20_30d)), top_20_30d['Forecast_30d'], color='steelblue')
axes[0, 0].set_yticks(range(len(top_20_30d)))
axes[0, 0].set_yticklabels([f"Store {i}" for i in top_20_30d['Store']])
axes[0, 0].set_xlabel('30-Day Forecast ($)', fontsize=12)
axes[0, 0].set_title('Top 20 Stores: 30-Day Cash Flow Forecast', fontweight='bold', fontsize=14)
axes[0, 0].grid(True, alpha=0.3, axis='x')

# 2. 90-day forecasts
top_20_90d = results_df.nlargest(20, 'Forecast_90d')
axes[0, 1].barh(range(len(top_20_90d)), top_20_90d['Forecast_90d'], color='coral')
axes[0, 1].set_yticks(range(len(top_20_90d)))
axes[0, 1].set_yticklabels([f"Store {i}" for i in top_20_90d['Store']])
axes[0, 1].set_xlabel('90-Day Forecast ($)', fontsize=12)
axes[0, 1].set_title('Top 20 Stores: 90-Day Cash Flow Forecast', fontweight='bold', fontsize=14)
axes[0, 1].grid(True, alpha=0.3, axis='x')

# 3. Forecast distribution
axes[1, 0].hist(results_df['Forecast_30d'], bins=30, color='lightblue', edgecolor='black', alpha=0.7)
axes[1, 0].axvline(results_df['Forecast_30d'].mean(), color='red', linestyle='--', 
                   label=f'Mean: ${results_df["Forecast_30d"].mean():,.0f}', linewidth=2)
axes[1, 0].set_xlabel('30-Day Forecast ($)')
axes[1, 0].set_ylabel('Number of Stores')
axes[1, 0].set_title('Distribution of 30-Day Forecasts Across All Stores', fontweight='bold', fontsize=14)
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')

# 4. Forecast by store type
store_types = results_df.groupby('StoreType').agg({
    'Forecast_30d': 'mean',
    'Forecast_90d': 'mean'
})

if len(store_types) > 0:
    x_pos = np.arange(len(store_types))
    width = 0.35
    
    axes[1, 1].bar(x_pos - width/2, store_types['Forecast_30d'], width, label='30-Day', alpha=0.7)
    axes[1, 1].bar(x_pos + width/2, store_types['Forecast_90d']/3, width, label='90-Day (÷3)', alpha=0.7)
    axes[1, 1].set_xlabel('Store Type', fontsize=12)
    axes[1, 1].set_ylabel('Average Forecast ($)')
    axes[1, 1].set_title('Forecast by Store Type', fontweight='bold', fontsize=14)
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels(store_types.index)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'multi_store_forecasts.png', dpi=300, bbox_inches='tight')
plt.close()

sample_stores = results_df.nlargest(10, 'Forecast_90d')
report_lines = []


for idx, row in sample_stores.iterrows():
    store_id = int(row['Store'])

    report_lines.append(f"Store {store_id} ({row['StoreType']}):")
    report_lines.append(f"  Historical average daily revenue: ${row['Historic_Avg']:,.2f}")
    report_lines.append(f"  Recent average daily revenue (30 days): ${row['Recent_Avg_Daily']:,.2f}")
    report_lines.append(f"  Predicted next day revenue: ${row['Predicted_Daily']:,.2f}")
    report_lines.append(f"  30-day revenue forecast: ${row['Forecast_30d']:,.0f}")
    report_lines.append(f"  60-day revenue forecast: ${row['Forecast_60d']:,.0f}")
    report_lines.append(f"  90-day revenue forecast: ${row['Forecast_90d']:,.0f}")
    report_lines.append("")

report_text = "\n".join(report_lines)
with open(output_dir / 'detailed_forecast_report.txt', 'w') as f:
    f.write(report_text)

print(" - Key Insights:")
if len(results_df) > 0:
    print(f"  - Average SME 30-day revenue forecast: ${results_df['Forecast_30d'].mean():,.0f}")
    print(f"  - Average SME 90-day revenue forecast: ${results_df['Forecast_90d'].mean():,.0f}")
    print(f"  - Best performing SME: ${results_df['Forecast_90d'].max():,.0f} (90 days)")