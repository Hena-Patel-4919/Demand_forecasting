import pandas as pd
import os

# =========================
# 1. Load Data
# =========================
file_path = "data/raw/retail_store_inventory.csv"
df = pd.read_csv(file_path)

# =========================
# 2. Convert Date
# =========================
df['Date'] = pd.to_datetime(df['Date'])

# =========================
# 3. Create Time Features
# =========================
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df['day_of_week'] = df['Date'].dt.dayofweek

# =========================
# 4. Remove Data Leakage
# =========================
df = df.drop(columns=['Demand Forecast'])

# =========================
# 5. Sort Data
# =========================
df = df.sort_values(by=['Store ID', 'Product ID', 'Date'])

# =========================
# 6. Save Processed Data
# =========================
output_path = "data/processed/cleaned_data.csv"
df.to_csv(output_path, index=False)

print("✅ Data saved to:", output_path)