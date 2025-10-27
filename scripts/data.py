import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# Load data
data_dir = Path("datasets")
train = pd.read_csv(data_dir / "train.csv")
store = pd.read_csv(data_dir / "store.csv")


# 1. Data Overview
print(f"   Train: {train.shape}")
print(f"   Store: {store.shape}")

# 2. Merge with store info
df = train.merge(store, on='Store', how='left')
df['Date'] = pd.to_datetime(df['Date'])
print(df.columns.tolist())

# 3. Check for missing values
print(df.isnull().sum())

# 4. Sales statistics (our "cash inflow")
print(df[df['Sales'] > 0]['Sales'].describe())

# 5. Time period
print(f"   Start: {df['Date'].min()}")
print(f"   End: {df['Date'].max()}")
print(f"   Total days: {(df['Date'].max() - df['Date'].min()).days}")

# 6. Sample time series for one store
store_1 = df[df['Store'] == 1].sort_values('Date')
store_1_open = store_1[store_1['Open'] == 1]

print(f"   Total records: {len(store_1)}")
print(f"   Open days: {store_1_open.shape[0]}")
print(f"   Closed days: {len(store_1) - store_1_open.shape[0]}")

# 7. Weekly patterns
weekly = df[df['Sales'] > 0].groupby('DayOfWeek')['Sales'].agg(['mean', 'std'])
print(weekly)

# 8. Save sample for visualization
output_dir = Path("outputs")
output_dir.mkdir(exist_ok=True)

# Sample for plotting (reduce data size)
sample_stores = df[df['Store'].isin([1, 2, 3, 4, 5])]
sample_stores.to_csv(output_dir / "reducedDataset.csv", index=False)


