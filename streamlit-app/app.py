# streamlit-app/app.py
import streamlit as st
import requests
import pandas as pd
import os

# Use environment variable for API URL (default: localhost)
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

# Load dataset from data/ folder
try:
    df = pd.read_csv("../data/creditcard.csv")
except FileNotFoundError:
    st.error("Dataset not found. Place creditcard.csv in the data/ folder.")
    st.stop()

st.title("Personal Finance Manager with Fraud Detection")

# Display sample data
with st.expander("View sample data"):
    st.dataframe(df[["Time", "Amount", "V1", "V2", "V3", "V4", "Class"]].head())

# Transaction form
with st.form("transaction_form"):
    st.caption("V1–V4 are anonymized PCA components. Use values between -30 and 30.")
    
    time = st.number_input("Time since first transaction (seconds)")
    amount = st.number_input("Amount", min_value=0.0)
    v1 = st.number_input("V1 (Anonymized Feature 1)", value=-1.359807)
    v2 = st.number_input("V2 (Anonymized Feature 2)", value=0.000000)
    v3 = st.number_input("V3 (Anonymized Feature 3)", value=1.773209)
    v4 = st.number_input("V4 (Anonymized Feature 4)", value=0.000000)
    
    if st.form_submit_button("Check Transaction"):
        transaction_data = {
            "time": time,
            "amount": amount,
            "v1": v1,
            "v2": v2,
            "v3": v3,
            "v4": v4
        }
        
        try:
            response = requests.post(API_URL, json=transaction_data)
            if response.status_code == 200:
                result = response.json()
                st.write(f"Fraud Detection: {'🚨 Fraud Detected' if result['fraud'] else '✅ Legitimate'}")
                st.write(f"Risk Probability: {result['probability']:.2%}")
            else:
                st.error(f"API Error {response.status_code}: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("Could not connect to the API. Is the FastAPI server running?")
