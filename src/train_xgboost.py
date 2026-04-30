
import pandas as pd
import numpy as np
import pickle
import os
import logging
from datetime import datetime
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import xgboost as xgb

# =========================
# LOGGING SETUP
# =========================
log_dir = r"C:\Users\Krish Patel\OneDrive\Desktop\M_tech\demand\reports"
os.makedirs(log_dir, exist_ok=True)

log_file = os.path.join(
    log_dir,
    f"training_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# File handler
file_handler = logging.FileHandler(log_file)
file_handler.setLevel(logging.INFO)

# Console handler (print in terminal)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Format
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers
logger.addHandler(file_handler)
logger.addHandler(console_handler)

logging.info("===== MODEL TRAINING STARTED =====")

# =========================
# 1. LOAD DATA
# =========================
file_path = r"C:\Users\Krish Patel\OneDrive\Desktop\M_tech\demand\data\processed\final_features.csv"
df = pd.read_csv(file_path)

print("Data Loaded:", df.shape)
logging.info(f"Data Loaded: {df.shape}")

# =========================
# 2. DROP LEAKAGE COLUMNS
# =========================
leakage_cols = [
    'Units Ordered',
    'Demand Forecast'
]

for col in leakage_cols:
    if col in df.columns:
        df = df.drop(columns=[col])

# =========================
# REMOVE LEAKY FEATURES
# =========================
suspicious_cols = [
    'inventory_gap',
    'order_gap',
    'rolling_mean_30'
]

df = df.drop(columns=[col for col in suspicious_cols if col in df.columns])

# =========================
# 3. DROP UNUSED RAW COLUMNS
# =========================
if 'Product ID' in df.columns:
    df = df.drop(columns=['Product ID'])

# =========================
# 4. HANDLE OBJECT COLUMNS
# =========================
for col in df.select_dtypes(include='object').columns:
    df[col] = df[col].astype('category')

# =========================
# 5. DEFINE TARGET
# =========================
target = 'Units Sold'

X = df.drop(columns=[target])
y = df[target]

# =========================
# 6. TRAIN TEST SPLIT
# =========================
df = df.sort_values(by=['year', 'month', 'day']).reset_index(drop=True)
split = int(len(df) * 0.8)

X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

print("Train size:", X_train.shape)
print("Test size:", X_test.shape)

logging.info(f"Train size: {X_train.shape}")
logging.info(f"Test size: {X_test.shape}")

# =========================
# 7. MODEL
# =========================
model = XGBRegressor(
    n_estimators=800,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1,
    random_state=42,
    enable_categorical=True
)

# =========================
# 8. TRAIN
# =========================
model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=100
)

print("Model training completed")
logging.info("Model training completed")

# =========================
# 9. EVALUATION
# =========================
preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
rmse = np.sqrt(mean_squared_error(y_test, preds))

print("\nModel Performance:")
print("MAE:", mae)
print("RMSE:", rmse)

logging.info("Model Performance:")
logging.info(f"MAE: {mae}")
logging.info(f"RMSE: {rmse}")

# =========================
# SAVE METRICS
# =========================
metrics_file = os.path.join(log_dir, "metrics.txt")

with open(metrics_file, "a") as f:
    f.write("\n===== NEW RUN =====\n")
    f.write(f"Timestamp: {datetime.now()}\n")
    f.write(f"MAE: {mae}\n")
    f.write(f"RMSE: {rmse}\n")

# =========================
# SAMPLE PREDICTIONS
# =========================
print("\nSample Predictions:")
pred_file = os.path.join(log_dir, "predictions_sample.txt")

with open(pred_file, "a") as f:
    f.write("\n===== SAMPLE PREDICTIONS =====\n")
    for i in range(min(10, len(preds))):
        print("Actual:", y_test.iloc[i], "Pred:", preds[i])
        f.write(f"Actual: {y_test.iloc[i]} | Pred: {preds[i]}\n")

# =========================
# 10. SAVE MODEL
# =========================
model_path = r"C:\Users\Krish Patel\OneDrive\Desktop\M_tech\demand\src\models\xgboost_model.pkl"

with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print("\nModel saved at:", model_path)
logging.info(f"Model saved at: {model_path}")

# =========================
# TARGET STATS
# =========================


logging.info("Target stats:")
logging.info("Target stats:\n" + str(y.describe()))

# =========================
# FEATURE IMPORTANCE
# =========================
plot_path = os.path.join(log_dir, "feature_importance.png")

xgb.plot_importance(model, max_num_features=15)
plt.savefig(plot_path)
plt.close()

logging.info(f"Feature importance plot saved at: {plot_path}")

logging.info("===== TRAINING COMPLETED =====")


# Naive baseline
baseline_preds = X_test['lag_1']

baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_preds))

print("\nBaseline RMSE:", baseline_rmse)
logging.info(f"Baseline RMSE: {baseline_rmse}")