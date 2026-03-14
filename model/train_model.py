import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

# Path to reference dataset (relative to project root)
DATA_PATH = os.path.join("data", "reference_data.csv")

# Check if dataset exists
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Reference dataset not found at {DATA_PATH}. "
                            f"Run 'python data/generate_data.py' first.")

# Load reference dataset
data = pd.read_csv(DATA_PATH)

# Use only the 3 features present in your dataset
X = data[["transaction_amount", "account_age_days", "num_transactions"]]
y = data["fraud"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Save model inside model/ folder
MODEL_PATH = os.path.join("model", "model.pkl")
joblib.dump(model, MODEL_PATH)

print(f"Model trained and saved at {MODEL_PATH}")