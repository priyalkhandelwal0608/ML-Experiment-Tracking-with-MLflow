from fastapi import FastAPI, Form
from fastapi.responses import HTMLResponse
import joblib
import os

# Resolve absolute path to model.pkl
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")

# Load trained model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model file not found at {MODEL_PATH}. "
        f"Run 'python model/train_model.py' first."
    )

model = joblib.load(MODEL_PATH)

app = FastAPI(title="Fraud Detection API")

@app.get("/", response_class=HTMLResponse)
def home():
    return """
    <html>
        <head>
            <title>Fraud Detection</title>
        </head>
        <body>
            <h2>Fraud Detection Form</h2>
            <form action="/predict" method="post">
                <label>Transaction Amount:</label>
                <input type="number" step="any" name="transaction_amount" required><br><br>
                
                <label>Account Age (days):</label>
                <input type="number" name="account_age_days" required><br><br>
                
                <label>Number of Transactions:</label>
                <input type="number" name="num_transactions" required><br><br>
                
                <button type="submit">Predict</button>
            </form>
        </body>
    </html>
    """

@app.post("/predict", response_class=HTMLResponse)
def predict(
    transaction_amount: float = Form(...),
    account_age_days: int = Form(...),
    num_transactions: int = Form(...)
):
    try:
        X = [[transaction_amount, account_age_days, num_transactions]]
        prediction = model.predict(X)[0]
        result = "Fraudulent Transaction" if prediction == 1 else "Legitimate Transaction"
    except Exception as e:
        result = f"Error: {str(e)}"

    return f"""
    <html>
        <head><title>Prediction Result</title></head>
        <body>
            <h2>Prediction Result</h2>
            <p><b>{result}</b></p>
            <a href="/">Go Back</a>
        </body>
    </html>
    """

# Allow running directly with python api/app.py
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)