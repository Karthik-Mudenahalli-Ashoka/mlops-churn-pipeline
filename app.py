from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np

app = FastAPI(title="Churn Prediction API", version="1.0")

# Load model and scaler
with open("models/best_model.pkl", "rb") as f:
    model = pickle.load(f)

# Input schema
class CustomerData(BaseModel):
    tenure: int
    monthly_charges: float
    total_charges: float
    contract_type: int
    payment_method: int
    tech_support: int
    online_security: int
    num_services: int

@app.get("/")
def home():
    return {"message": "Churn Prediction API is running!", "version": "1.0"}

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(customer: CustomerData):
    features = np.array([[
        customer.tenure,
        customer.monthly_charges,
        customer.total_charges,
        customer.contract_type,
        customer.payment_method,
        customer.tech_support,
        customer.online_security,
        customer.num_services
    ]])
    
    prediction = model.predict(features)[0]
    probability = model.predict_proba(features)[0][1]
    
    return {
        "churn_prediction": int(prediction),
        "churn_probability": round(float(probability), 4),
        "risk_level": "High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low"
    }