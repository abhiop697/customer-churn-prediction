import joblib
import numpy as np
import pandas as pd

# Load trained model
model = joblib.load("F:/project/customer_churn/models/churn_model.pkl")

# Define actual feature names before encoding
feature_names = [
    'Age', 'Gender', 'Tenure', 'Usage Frequency', 'Support Calls', 
    'Payment Delay', 'Subscription Type', 'Contract Length', 
    'Total Spend', 'Last Interaction'
]

# Example new customer data (ensure 10 features before encoding)
sample_customer = pd.DataFrame([[
    35, 'Male', 24, 3, 0, 2, 'Premium', 'Monthly', 500, 12
]], columns=feature_names)

# One-hot encode categorical variables
sample_customer = pd.get_dummies(sample_customer)

# Get the expected feature names from the trained model
expected_features = model.feature_names_in_

# Ensure the sample customer data has the same features as the model
for feature in expected_features:
    if feature not in sample_customer.columns:
        sample_customer[feature] = 0  # Add missing columns with 0 value

# Reorder columns to match the model's training data
sample_customer = sample_customer[expected_features]

# Predict churn
prediction = model.predict(sample_customer)
print("Predicted Churn:", "Yes" if prediction[0] == 1 else "No")
