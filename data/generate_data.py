import pandas as pd
import numpy as np

np.random.seed(42)

n = 5000

data = pd.DataFrame({
    "transaction_amount": np.random.normal(100, 20, n),
    "account_age_days": np.random.normal(500, 100, n),
    "num_transactions": np.random.normal(5, 2, n),
})

data["fraud"] = (
    (data["transaction_amount"] > 140) &
    (data["num_transactions"] > 7)
).astype(int)

data.to_csv("reference_data.csv", index=False)

print("Reference dataset created")