import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

def retrain():
    # Paths relative to project root
    REFERENCE = os.path.join("data", "reference_data.csv")
    MODEL_PATH = os.path.join("model", "model.pkl")

    # Check if reference dataset exists
    if not os.path.exists(REFERENCE):
        raise FileNotFoundError(
            f"Reference dataset not found at {REFERENCE}. "
            f"Run 'python data/generate_data.py' first."
        )

    # Load reference dataset
    ref = pd.read_csv(REFERENCE)

    # Features and target
    if "fraud" not in ref.columns:
        raise ValueError("Reference dataset must contain a 'fraud' column as target.")

    X = ref[["transaction_amount", "account_age_days", "num_transactions"]]
    y = ref["fraud"]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Retrain model
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Save updated model
    joblib.dump(model, MODEL_PATH)
    print(f"Model retrained and saved at {MODEL_PATH}")

if __name__ == "__main__":
    retrain()