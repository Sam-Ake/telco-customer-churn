from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
from pydantic import BaseModel

# Initialize FastAPI app
app = FastAPI(
    title="Telco Customer Churn Prediction API",
    description="A lightweight REST API for predicting customer churn using a trained machine learning pipeline.",
    version="1.0",
)

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the trained pipeline saved from Kaggle
model = joblib.load("churn_pipeline.pkl")

# Define Pydantic model for incoming JSON payload
class CustomerData(BaseModel):
    gender: str
    SeniorCitizen: int
    Partner: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    MonthlyCharges: float
    TotalCharges: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Telco Customer Churn Prediction API!"}

@app.post("/churn")
def predict_churn(data: CustomerData):
    input_data = pd.DataFrame([data.dict()])
    input_data['TotalCharges'] = pd.to_numeric(input_data['TotalCharges'], errors='coerce').fillna(0)
    input_data['ChargePerTenure'] = input_data['TotalCharges'] / (input_data['tenure'] + 1)
    
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]
    
    result = "Yes" if prediction == 1 else "No"
    
    return {
        "churn_prediction": result,
        "churn_probability": round(float(probability), 4),
        "risk_level": "High" if probability > 0.6 else ("Moderate" if probability > 0.3 else "Low")
    }
