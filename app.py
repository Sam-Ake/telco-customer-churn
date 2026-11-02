import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# Page Configuration
st.set_page_config(
    page_title="Telco Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# Load the trained model pipeline
@st.cache_resource
def load_model():
    return joblib.load("churn_pipeline.pkl")

model = load_model()

# App Header
st.title("📞 Telco Customer Churn Prediction App")
st.markdown("Predict whether a customer is likely to churn based on their account info, service subscriptions, and billing data.")

# Sidebar for User Inputs
st.sidebar.header("Customer Details")

def user_input_features():
    # Demographics
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    SeniorCitizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
    Partner = st.sidebar.selectbox("Has Partner", ["Yes", "No"])
    Dependents = st.sidebar.selectbox("Has Dependents", ["Yes", "No"])
    
    # Account Info
    tenure = st.sidebar.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
    Contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    PaperlessBilling = st.sidebar.selectbox("Paperless Billing", ["Yes", "No"])
    PaymentMethod = st.sidebar.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    
    # Financials
    MonthlyCharges = st.sidebar.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=70.0)
    TotalCharges = st.sidebar.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=800.0)
    
    # Services
    PhoneService = st.sidebar.selectbox("Phone Service", ["Yes", "No"])
    MultipleLines = st.sidebar.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    InternetService = st.sidebar.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    OnlineSecurity = st.sidebar.selectbox("Online Security", ["Yes", "No", "No internet service"])
    OnlineBackup = st.sidebar.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    DeviceProtection = st.sidebar.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    TechSupport = st.sidebar.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    StreamingTV = st.sidebar.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    StreamingMovies = st.sidebar.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    
    # Compile into dictionary
    data = {
        "gender": gender,
        "SeniorCitizen": SeniorCitizen,
        "Partner": Partner,
        "Dependents": Dependents,
        "tenure": tenure,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges
    }
    
    return pd.DataFrame([data])

input_df = user_input_features()

# Main Panel Display
st.subheader("Customer Profile Summary")
st.dataframe(input_df)

# Prediction Button
if st.button("Predict Churn"):
    # Apply feature engineering matching training
    input_df['TotalCharges'] = pd.to_numeric(input_df['TotalCharges'], errors='coerce').fillna(0)
    input_df['ChargePerTenure'] = input_df['TotalCharges'] / (input_df['tenure'] + 1)
    
    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Prediction Result")
        if prediction == 1:
            st.error("⚠️ **High Risk: This customer is likely to churn!**")
        else:
            st.success("✅ **Low Risk: This customer is likely to stay.**")
            
        st.write(f"**Churn Probability:** `{probability * 100:.2f}%`")

    with col2:
        # Plotly Gauge Chart for Risk Visualization
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = probability * 100,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Churn Risk Score", 'font': {'size': 24}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 30], 'color': "lightgreen"},
                    {'range': [30, 60], 'color': "yellow"},
                    {'range': [60, 100], 'color': "salmon"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 60
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)
