import pandas as pd
import os

# =========================
# 1. Define paths
# =========================
BASE_PATH = os.getcwd()

RAW_PATH = os.path.join(BASE_PATH, "data", "raw", "retail_store_inventory.csv")
PROCESSED_PATH = os.path.join(BASE_PATH, "data", "processed", "final_data.csv")

# =========================
# 2. Load Data
# =========================
df = pd.read_csv(RAW_PATH)

# =========================
# 3. Fix Category Mapping
# =========================
product_category_map = (
    df.groupby('Product ID')['Category']
    .agg(lambda x: x.mode()[0])
    .to_dict()
)

df['Category'] = df['Product ID'].map(product_category_map)

# =========================
# 4. Convert Date
# =========================
df['Date'] = pd.to_datetime(df['Date'])

# =========================
# 5. Sort Data
# =========================
df = df.sort_values(by=['Store ID', 'Product ID', 'Date'])

# =========================
# 6. Create Time Features
# =========================
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df['day_of_week'] = df['Date'].dt.dayofweek

# =========================
# 7. Create Lag Features
# =========================
df['lag_1'] = df.groupby(['Store ID', 'Product ID'])['Units Sold'].shift(1)
df['lag_7'] = df.groupby(['Store ID', 'Product ID'])['Units Sold'].shift(7)

# =========================
# 8. Rolling Features
# =========================
df['rolling_mean_7'] = (
    df.groupby(['Store ID', 'Product ID'])['Units Sold']
    .shift(1)
    .rolling(7)
    .mean()
)

# =========================
# 9. Drop NA values
# =========================
df = df.dropna()

# =========================
# 10. Save processed data
# =========================
os.makedirs(os.path.dirname(PROCESSED_PATH), exist_ok=True)
df.to_csv(PROCESSED_PATH, index=False)

print("Processed data saved at:", PROCESSED_PATH)
print("Final shape:", df.shape)