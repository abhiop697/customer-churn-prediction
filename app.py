import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("models/churn_model.pkl")

# Streamlit UI
st.title(" Customer Churn Prediction")
st.write("Enter customer details to predict churn.")

# User inputs (Ensure these match your dataset features)
age = st.number_input("Age", 18, 100)
tenure = st.number_input("Tenure (Months)", 0, 72)
monthly_charges = st.number_input("Monthly Charges ($)", 0, 500)
total_charges = st.number_input("Total Charges ($)", 0, 10000)
contract_type = st.selectbox("Contract Type", ["Month-to-Month", "One Year", "Two Year"])
payment_method = st.selectbox("Payment Method", ["Credit Card", "Bank Transfer", "Electronic Check", "Mailed Check"])
internet_service = st.selectbox("Internet Service", ["DSL", "Fiber Optic", "No Internet"])
online_security = st.selectbox("Online Security", ["Yes", "No"])
tech_support = st.selectbox("Tech Support", ["Yes", "No"])
streaming_tv = st.selectbox("Streaming TV", ["Yes", "No"])
streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No"])
paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

# Encode categorical features (Ensure these match your dataset)
contract_mapping = {"Month-to-Month": 0, "One Year": 1, "Two Year": 2}
payment_mapping = {"Credit Card": 0, "Bank Transfer": 1, "Electronic Check": 2, "Mailed Check": 3}
internet_mapping = {"DSL": 0, "Fiber Optic": 1, "No Internet": 2}
binary_mapping = {"Yes": 1, "No": 0}

contract_type = contract_mapping[contract_type]
payment_method = payment_mapping[payment_method]
internet_service = internet_mapping[internet_service]
online_security = binary_mapping[online_security]
tech_support = binary_mapping[tech_support]
streaming_tv = binary_mapping[streaming_tv]
streaming_movies = binary_mapping[streaming_movies]
paperless_billing = binary_mapping[paperless_billing]

# Ensure the number of features is correct (12 features)
sample_data = np.array([
    age, tenure, monthly_charges, total_charges, contract_type, payment_method,
    internet_service, online_security, tech_support, streaming_tv, streaming_movies, paperless_billing
]).reshape(1, -1)

# Predict button
if st.button("Predict Churn"):
    prediction = model.predict(sample_data)
    result = "Yes" if prediction[0] == 1 else "No"
    st.success(f"Prediction: **{result}**")
