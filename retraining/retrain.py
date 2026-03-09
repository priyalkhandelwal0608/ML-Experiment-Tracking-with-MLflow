import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

REFERENCE = "../data/reference_data.csv"
PRODUCTION = "../data/production_data.csv"

def retrain():

    ref = pd.read_csv(REFERENCE)
    prod = pd.read_csv(PRODUCTION)

    prod["fraud"] = prod["prediction"]

    new_data = pd.concat([ref, prod])

    X = new_data.drop("fraud", axis=1)
    y = new_data["fraud"]

    model = RandomForestClassifier()

    model.fit(X, y)

    joblib.dump(model, "../model/model.pkl")

    print("Model retrained and updated")


if __name__ == "__main__":
    retrain()