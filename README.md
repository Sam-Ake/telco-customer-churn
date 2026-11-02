# Telco Customer Churn Prediction & Deployment Pipeline

An end-to-end machine learning project predicting customer churn for a telecommunications company. This project covers the full lifecycle from data preprocessing and feature engineering to REST API development and interactive frontend deployment.

## Project Architecture
- **Model Training & Validation (`notebooks/`)**: Built using Python, Pandas, Scikit-learn, and Stratified K-Fold cross-validation to handle class imbalance, evaluated via Recall, F1-Score, and ROC-AUC.
- **Backend API (`main.py`)**: A lightweight FastAPI application providing a `/churn` prediction endpoint with Pydantic validation.
- **Frontend App (`app.py`)**: An interactive Streamlit dashboard featuring custom customer profile inputs and dynamic Plotly risk gauge visualizations.

## Key Features
- **Feature Engineering**: Engineered a custom `ChargePerTenure` interaction feature to capture monthly spending relative to customer lifetime.
- **Robust Pipeline**: Automated preprocessing using `ColumnTransformer` (Standard Scaling for numeric data, One-Hot Encoding for categoricals).
- **Production Ready**: Serialized using `joblib` and structured for seamless deployment.

## Tech Stack
- **Languages**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, Imbalanced-learn, Joblib
- **Backend**: FastAPI, Uvicorn
- **Frontend**: Streamlit, Plotly
- **Version Control & Hosting**: GitHub, Streamlit Cloud / Vercel
