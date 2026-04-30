import pandas as pd
import numpy as np
import os

# =========================
# 1. LOAD DATA
# =========================
file_path = r"C:\Users\Krish Patel\OneDrive\Desktop\M_tech\demand\data\raw\retail_store_inventory.csv"
df = pd.read_csv(file_path)

print("Data Loaded:", df.shape)

# =========================
# 2. DATE PROCESSING
# =========================
df['Date'] = pd.to_datetime(df['Date'])

df = df.sort_values(by=['Store ID', 'Product ID', 'Date'])

df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df['day_of_week'] = df['Date'].dt.dayofweek
df['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)

df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# =========================
# 3. TARGET ENCODING (PRODUCT)
# =========================
product_mean = df.groupby('Product ID')['Units Sold'].mean()
df['product_encoded'] = df['Product ID'].map(product_mean)

# =========================
# 4. ONE HOT ENCODING (STORE)
# =========================
df = pd.get_dummies(df, columns=['Store ID'], drop_first=True)

# =========================
# 5. LAG FEATURES
# =========================
df = df.sort_values(by=['Product ID', 'year', 'month', 'day'])

df['lag_1'] = df.groupby('Product ID')['Units Sold'].shift(1)
df['lag_7'] = df.groupby('Product ID')['Units Sold'].shift(7)
df['lag_14'] = df.groupby('Product ID')['Units Sold'].shift(14)
df['lag_30'] = df.groupby('Product ID')['Units Sold'].shift(30)

# =========================
# 6. ROLLING FEATURES
# =========================
df['rolling_mean_7'] = df.groupby('Product ID')['Units Sold'].transform(lambda x: x.shift(1).rolling(7).mean())
df['rolling_std_7'] = df.groupby('Product ID')['Units Sold'].transform(lambda x: x.shift(1).rolling(7).std())

df['rolling_mean_14'] = df.groupby('Product ID')['Units Sold'].transform(lambda x: x.shift(1).rolling(14).mean())
df['rolling_mean_30'] = df.groupby('Product ID')['Units Sold'].transform(lambda x: x.shift(1).rolling(30).mean())

# =========================
# 7. PRICE BASED FEATURES
# =========================
df['price_discount'] = df['Price'] * df['Discount']
df['price_competitor_diff'] = df['Price'] - df['Competitor Pricing']

# =========================
# 8. DEMAND PRESSURE FEATURES
# =========================
df['inventory_gap'] = df['Inventory Level'] - df['Units Sold']
df['order_gap'] = df['Units Ordered'] - df['Units Sold']

# =========================
# 9. DROP UNUSED / LEAKY COLUMNS
# =========================
df = df.drop(columns=[
    'Date',
    'Category',
    'Region',
    'Weather Condition',
    'Seasonality',
    'Demand Forecast'  # IMPORTANT: avoid leakage
])

# =========================
# 10. HANDLE MISSING VALUES (FROM LAGS)
# =========================
df = df.dropna()

# =========================
# 11. SAVE PROCESSED DATA
# =========================
output_path = r"C:\\Users\\Krish Patel\\OneDrive\\Desktop\\M_tech\\demand\\data\\processed\\final_features.csv"
df.to_csv(output_path, index=False)

print("Feature engineering completed")
print("Final shape:", df.shape)
print("Saved at:", output_path)