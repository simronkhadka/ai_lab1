import streamlit as st
import joblib
import pandas as pd

st.title("🎯 Customer Churn Prediction")

model = joblib.load('knn_model.joblib')
scaler = joblib.load('scaler.joblib')

st.subheader("Enter Customer Details:")

monthly_charges = st.number_input(
    "Monthly Charges ($)", 
    min_value=20.0, 
    max_value=120.0, 
    value=70.0,
    step=5.0
)

tenure_months = st.number_input(
    "Tenure (Months)", 
    min_value=1, 
    max_value=72, 
    value=12,
    step=1
)

total_charges = st.number_input(
    "Total Charges ($)", 
    min_value=20.0, 
    max_value=10000.0, 
    value=840.0,
    step=50.0
)

service_calls = st.number_input(
    "Service Calls", 
    min_value=0, 
    max_value=10, 
    value=2,
    step=1
)

contract_type = st.selectbox(
    "Contract Type",
    options=[0, 1, 2],
    format_func=lambda x: ["Month-to-Month", "One Year", "Two Year"][x]
)

# Create sample input
sample = [[monthly_charges, tenure_months, total_charges, service_calls, contract_type]]

# Churn mapping
churn_map = {
    0: "Customer will STAY",
    1: "Customer will CHURN"
}

# Predict button
if st.button("Predict"):
    # Scale the input
    sample_scaled = scaler.transform(sample)
    
    # Make prediction
    prediction = model.predict(sample_scaled)[0]
    
    # Display result
    churn_status = churn_map[prediction]
    
    if prediction == 1:
        st.error(f"⚠️ {churn_status}")
    else:
        st.success(f"✅ {churn_status}")
