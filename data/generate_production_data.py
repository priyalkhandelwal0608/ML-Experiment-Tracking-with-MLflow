import pandas as pd
import numpy as np
import joblib
import os

# Resolve path to model.pkl relative to project root
MODEL_PATH = os.path.join("model", "model.pkl")

# Load the trained model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. "
                            f"Run 'python model/train_model.py' first.")

model = joblib.load(MODEL_PATH)

# Set random seed for reproducibility
np.random.seed(123)
n = 2000

# Simulate production data with shifted distributions (to mimic drift)
prod_data = pd.DataFrame({
    "transaction_amount": np.random.normal(120, 25, n),   # shifted mean
    "account_age_days": np.random.normal(480, 120, n),    # slightly different
    "num_transactions": np.random.normal(6, 2.5, n),      # changed variance
})

# Generate predictions using the trained model
X = prod_data[["transaction_amount", "account_age_days", "num_transactions"]]
prod_data["prediction"] = model.predict(X)

# Save production dataset
OUTPUT_PATH = os.path.join("data", "production_data.csv")
prod_data.to_csv(OUTPUT_PATH, index=False)
print(f"Production dataset created at {OUTPUT_PATH}")