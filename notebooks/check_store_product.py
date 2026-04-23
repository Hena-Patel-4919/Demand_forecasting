import pandas as pd

# =========================
# 1. Load Data
# =========================
df = pd.read_csv("data/raw/retail_store_inventory.csv")

# =========================
# 2. FIX CATEGORY MAPPING
# =========================
# Get most frequent category per product
product_category_map = (
    df.groupby('Product ID')['Category']
    .agg(lambda x: x.mode()[0])
)

# Convert to dictionary
product_category_map = product_category_map.to_dict()

# Apply mapping (IMPORTANT: overwrite column)
df['Category'] = df['Product ID'].map(product_category_map)

print("Category inconsistency fixed")

# =========================
# 3. VERIFY FIX
# =========================
check = df.groupby('Product ID')['Category'].nunique()

print("\nCategory uniqueness per product:")
print(check.head(10))

# =========================
# 4. FINAL REQUIRED OUTPUT
# =========================
# Unique Store-Product-Category mapping
final_df = df[['Store ID', 'Product ID', 'Category']].drop_duplicates()

# Sort for clean view
final_df = final_df.sort_values(by=['Store ID', 'Product ID'])

print("\nFINAL OUTPUT (Clean Mapping):")
print(final_df.head(50))