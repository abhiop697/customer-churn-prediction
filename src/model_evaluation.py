import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load trained model
model = joblib.load("models/churn_model.pkl")

# Load training data
X_train = pd.read_csv("data/X_train.csv")

# Feature importance
feature_importance = model.feature_importances_
feature_names = X_train.columns

# Plot feature importance
plt.figure(figsize=(10, 5))
plt.barh(feature_names, feature_importance)
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance in Churn Prediction")
plt.show()
