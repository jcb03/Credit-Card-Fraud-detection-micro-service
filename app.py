import streamlit as st
import requests
import pandas as pd

API_URL = "http://api:8000/predict"

st.title("Personal Finance Manager with Fraud Detection")

with st.form("transaction_form"):
    time = st.number_input("Time since first transaction (seconds)")
    amount = st.number_input("Amount", min_value=0.0)
    v1 = st.number_input("V1 (Anonymized Feature 1)")
    v2 = st.number_input("V2 (Anonymized Feature 2)")
    v3 = st.number_input("V3 (Anonymized Feature 3)")
    v4 = st.number_input("V4 (Anonymized Feature 4)")
    
    if st.form_submit_button("Check Transaction"):
        transaction_data = {
            "time": time,
            "amount": amount,
            "v1": v1,
            "v2": v2,
            "v3": v3,
            "v4": v4
        }
        
        response = requests.post(API_URL, json=transaction_data)
        if response.status_code == 200:
            result = response.json()
            st.write(f"Fraud Detection: {'🚨 Fraud Detected' if result['fraud'] else '✅ Legitimate'}")
            st.write(f"Risk Probability: {result['probability']:.2%}")
        else:
            st.error("Error processing transaction")
