import joblib
import pandas as pd

# Load the trained model
model = joblib.load("models/churn_model.pkl")

# Load training data to check number of features
X_train = pd.read_csv("data/X_train.csv")

print("Model was trained on", X_train.shape[1], "features.")
