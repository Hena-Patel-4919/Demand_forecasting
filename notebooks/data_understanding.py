import pandas as pd
import os

print("Current Working Directory:", os.getcwd())
print("Files in data folder:", os.listdir("data"))
print("Files in raw folder:", os.listdir("data/raw"))

file_path = "data/raw/retail_store_inventory.csv"
df = pd.read_csv(file_path)

print(df.head())
# =========================
# 4. Basic Inspection
# =========================
print("\nShape of data:", df.shape)

print("\nColumns:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())

print("\nData Info:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

# =========================
# 5. Missing Values
# =========================
print("\nMissing Values:")
print(df.isnull().sum())

# =========================
# 6. Convert Date
# =========================
df['Date'] = pd.to_datetime(df['Date'])

# =========================
# 7. Sort Data
# =========================
df = df.sort_values(by='Date')

# =========================
# 8. Reset Index
# =========================
df = df.reset_index(drop=True)

print("\nFinal Data Preview:")
print(df.head())