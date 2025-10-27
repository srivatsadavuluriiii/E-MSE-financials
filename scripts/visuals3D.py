import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 12)

data_dir = Path("datasets")
train = pd.read_csv(data_dir / "train.csv", low_memory=False)
store = pd.read_csv(data_dir / "store.csv")

output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

# Prepare data
df = train.merge(store, on='Store', how='left')
df['Date'] = pd.to_datetime(df['Date'])
df = df[df['Open'] == 1].copy()


df['YearMonth'] = df['Date'].dt.to_period('M').astype(str)
df['Month'] = df['Date'].dt.month

# Get monthly averages by store type
monthly_by_type = df.groupby(['StoreType', 'Month'])['Sales'].mean().reset_index()

# Create pivot table for 3D plot
pivot_data = monthly_by_type.pivot(index='StoreType', columns='Month', values='Sales')

fig = plt.figure(figsize=(16, 12))
ax = fig.add_subplot(2, 2, 1, projection='3d')

# Prepare data for surface plot
store_types = pivot_data.index.values
months = pivot_data.columns.values
X, Y = np.meshgrid(range(len(months)), range(len(store_types)))
Z = pivot_data.values

# Create surface
surf = ax.plot_surface(X, Y, Z, cmap=cm.viridis, alpha=0.7, edgecolor='none')
ax.set_xlabel('Month (1-12)', fontsize=12)
ax.set_ylabel('Store Type', fontsize=12)
ax.set_zlabel('Average Cash Flow ($)', fontsize=12)
ax.set_title('Cash Flow Surface: Store Type vs Month', fontsize=14, fontweight='bold')
ax.set_xticks(range(len(months)))
ax.set_xticklabels(months)
ax.set_yticks(range(len(store_types)))
ax.set_yticklabels(store_types)
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)


scatter_df = df.sample(n=5000, random_state=42)  # Sample for performance

ax = fig.add_subplot(2, 2, 2, projection='3d')

# Group by DayOfWeek and Promo
grouped = df.groupby(['DayOfWeek', 'Promo'])['Sales'].mean().reset_index()

x = grouped['DayOfWeek'].values
y = grouped['Promo'].values * 1.5  # Spread out promo
z = grouped['Sales'].values

# Color by sales amount
colors = cm.viridis(z / z.max())

scatter = ax.scatter(x, y, z, c=z, cmap=cm.viridis, s=100, alpha=0.6, edgecolors='black', linewidths=0.5)
ax.set_xlabel('Day of Week (1=Mon, 7=Sun)', fontsize=12)
ax.set_ylabel('Promotion Active (0=No, 1=Yes)', fontsize=12)
ax.set_zlabel('Average Cash Flow ($)', fontsize=12)
ax.set_title('Promotion Effect: Day of Week vs Promo vs Cash Flow', fontsize=14, fontweight='bold')
ax.set_xticks(range(1, 8))
ax.set_yticks([0, 1.5])
ax.set_yticklabels(['No Promo', 'Promo'])
plt.colorbar(scatter, ax=ax, shrink=0.6, label='Cash Flow ($)')


try:
    forecasts = pd.read_csv(output_dir / 'all_stores_forecasts.csv')
    
    # Get top stores
    top_stores = forecasts.nlargest(30, 'Forecast_90d')
    
    ax = fig.add_subplot(2, 2, 3, projection='3d')
    
    x = top_stores['Store'].values
    y = [1] * len(x)  # Category identifier
    z = [0] * len(x)   # Base at 0
    dx = [1] * len(x)
    dy = [1] * len(x)
    dz = (top_stores['Forecast_90d'] / 1000).values  # Divide by 1000 for scale
    
    # Color by store type
    colors_map = {'a': 'blue', 'b': 'green', 'c': 'red', 'd': 'orange'}
    top_stores['color'] = top_stores['StoreType'].map(colors_map)
    colors = [colors_map.get(t, 'gray') for t in top_stores['StoreType']]
    
    ax.bar3d(x, y, z, dx, dy, dz, alpha=0.7, color=colors, edgecolor='black')
    ax.set_xlabel('Store ID', fontsize=12)
    ax.set_ylabel('Category', fontsize=12)
    ax.set_zlabel('90-Day Forecast (in $1000s)', fontsize=12)
    ax.set_title('Top 30 Stores: 90-Day Forecast by Store Type', fontsize=14, fontweight='bold')
    ax.set_yticks([1])
    ax.set_yticklabels(['High Performers'])
    
except Exception as e:
    print(f"   ⚠ Could not load forecasts: {e}")
    ax = fig.add_subplot(2, 2, 3, projection='3d')
    ax.text(0.5, 0.5, 0.5, 'Run 05_predict_all_stores.py\nfirst to generate forecast data', 
            horizontalalignment='center', fontsize=12)
    ax.set_title('Forecast Data Not Available', fontsize=14)
    ax.set_axis_off()


# Monthly sales by store type and year
monthly_data = df.groupby(['Month', 'StoreType'])['Sales'].mean().reset_index()
pivot_monthly = monthly_data.pivot(index='Month', columns='StoreType', values='Sales')

ax = fig.add_subplot(2, 2, 4, projection='3d')

X_month, Y_type = np.meshgrid(pivot_monthly.index.values, range(len(pivot_monthly.columns)))
Z_month = pivot_monthly.values.T

wire = ax.plot_wireframe(X_month, Y_type, Z_month, alpha=0.6, cmap=cm.coolwarm, linewidth=1)
ax.set_xlabel('Month (1-12)', fontsize=12)
ax.set_ylabel('Store Type', fontsize=12)
ax.set_zlabel('Average Cash Flow ($)', fontsize=12)
ax.set_title('Seasonal Cash Flow Patterns by Store Type', fontsize=14, fontweight='bold')
ax.set_xticks(pivot_monthly.index.values[::2])
ax.set_yticks(range(len(pivot_monthly.columns)))
ax.set_yticklabels(pivot_monthly.columns)

plt.tight_layout()
plt.savefig(output_dir / '3d_visualizations.png', dpi=300, bbox_inches='tight')
plt.close()


# Create simulated confidence intervals for visualization
sample_stores = df['Store'].value_counts().nlargest(30).index.values
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

forecasts_list = []
for store_id in sample_stores[:20]:
    store_data = df[df['Store'] == store_id]['Sales']
    if len(store_data) > 30:
        recent_avg = store_data.tail(30).mean()
        recent_std = store_data.tail(30).std()
        forecast = recent_avg
        
        forecasts_list.append({
            'store': store_id,
            'forecast': forecast,
            'lower': forecast - (recent_std * 1.96),
            'upper': forecast + (recent_std * 1.96),
            'std': recent_std
        })

if forecasts_list:
    forecast_df = pd.DataFrame(forecasts_list)
    
    x = forecast_df['store'].values
    y = forecast_df['forecast'].values
    z = forecast_df['std'].values  # Uncertainty
    
    # Size based on range
    sizes = (forecast_df['upper'] - forecast_df['lower']).values
    sizes_scaled = (sizes / sizes.max()) * 200
    
    scatter = ax.scatter(x, y, z, c=z, s=sizes_scaled, cmap=cm.plasma, 
                         alpha=0.6, edgecolors='black', linewidths=1)
    
    ax.set_xlabel('Store ID', fontsize=12)
    ax.set_ylabel('Predicted Cash Flow ($)', fontsize=12)
    ax.set_zlabel('Uncertainty (Std Dev)', fontsize=12)
    ax.set_title('Forecast Confidence: Store vs Prediction vs Uncertainty', 
                 fontsize=14, fontweight='bold')
    plt.colorbar(scatter, ax=ax, shrink=0.8, label='Uncertainty')
    
    # Add confidence intervals as lines
    for idx, row in forecast_df.iterrows():
        ax.plot([row['store'], row['store']], 
                [row['lower'], row['upper']], 
                [row['std'], row['std']], 
                'b-', alpha=0.3, linewidth=2)

plt.tight_layout()
plt.savefig(output_dir / '3d_forecast_confidence.png', dpi=300, bbox_inches='tight')
plt.close()

# Sample 3 stores over time
fig = plt.figure(figsize=(16, 10))
ax = fig.add_subplot(111, projection='3d')

stores_sample = [1, 2, 3]
colors_3d = ['blue', 'green', 'red']

for idx, store_id in enumerate(stores_sample):
    store_data = df[df['Store'] == store_id].sort_values('Date')
    if len(store_data) > 100:
        store_data = store_data.iloc[::10]  # Sample every 10th day
        
        dates_num = (store_data['Date'] - store_data['Date'].min()).dt.days
        x = dates_num.values
        y = [store_id] * len(x)
        z = store_data['Sales'].values
        
        ax.plot(x, y, z, label=f'Store {store_id}', color=colors_3d[idx], 
               linewidth=2, alpha=0.7)
        ax.scatter(x, y, z, color=colors_3d[idx], s=20, alpha=0.5, edgecolors='black')

ax.set_xlabel('Days Since Start', fontsize=12)
ax.set_ylabel('Store ID', fontsize=12)
ax.set_zlabel('Cash Flow ($)', fontsize=12)
ax.set_title('3D Cash Flow Timeline: Sample Stores Over Time', fontsize=14, fontweight='bold')
ax.legend()
ax.set_yticks(stores_sample)

plt.tight_layout()
plt.savefig(output_dir / '3d_timeline.png', dpi=300, bbox_inches='tight')
plt.close()
