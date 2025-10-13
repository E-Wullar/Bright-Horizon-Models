import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("cltv_model_filtered.pkl")

st.title("Customer Lifetime Value (CLTV) Predictor")

st.markdown("### Enter Customer Information")

# Input fields (only impactful ones)
total_profit = st.number_input("Total Profit", min_value=0.0, format="%.2f")
annualized_deposits = st.number_input("Annualized Deposits", min_value=0.0, format="%.2f")
balance_to_income = st.number_input("Balance to Income Ratio", min_value=0.0, format="%.2f")
monthly_tx_rate = st.number_input("Monthly Transaction Rate", min_value=0.0, format="%.2f")
default_risk_score = st.slider("Default Risk Score", min_value=0, max_value=100)
monthly_income = st.number_input("Monthly Income (GHS)", min_value=0.0, format="%.2f")

employment_sector = st.selectbox("Employment Sector", ["Public", "Private", "Self-Employed", "Unemployed", "Other"])
employment_mapping = {
    "Public": 0,
    "Private": 1,
    "Self-Employed": 2,
    "Unemployed": 3,
    "Other": 4
}
employment_sector_encoded = employment_mapping[employment_sector]

# Build input dataframe
input_data = pd.DataFrame([[
    total_profit,
    annualized_deposits,
    balance_to_income,
    monthly_tx_rate,
    default_risk_score,
    monthly_income,
    employment_sector_encoded
]], columns=[
    'Total_Profit',
    'Annualized_Deposits',
    'Balance_to_Income_Ratio',
    'Monthly_Transaction_Rate',
    'Default_Risk_Score',
    'Monthly_Income_(GHS)',
    'Employment_Sector'
])

# Predict and show result
if st.button("Predict CLTV"):
    prediction = model.predict(input_data)
    st.success(f"Estimated Customer Lifetime Value: GHS {prediction[0]:,.2f}")
