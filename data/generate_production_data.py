import pandas as pd
import numpy as np
import joblib

# Load the trained model to generate predictions
model = joblib.load("../model/model.pkl")

np.random.seed(123)
n = 2000

# Simulate production data with slightly shifted distributions (to mimic drift)
prod_data = pd.DataFrame({
    "transaction_amount": np.random.normal(120, 25, n),  # shifted mean
    "account_age_days": np.random.normal(480, 120, n),   # slightly different
    "num_transactions": np.random.normal(6, 2.5, n),     # changed variance
})

# Use the trained model to generate predictions
X = prod_data[["transaction_amount", "account_age_days", "num_transactions"]]
prod_data["prediction"] = model.predict(X)

# Save production dataset
prod_data.to_csv("production_data.csv", index=False)
print("Production dataset created")