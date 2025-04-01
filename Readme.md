# Customer Churn Prediction  
Predicting customer churn using Machine Learning & Data Analysis  
---

## Project Overview  
Customer churn is a critical problem for businesses, especially in industries like telecom, banking, and e-commerce. This project builds a machine learning model to predict whether a customer is likely to churn based on historical data.  

The project includes:  
Data Preprocessing & Cleaning  
Feature Engineering & Selection  
Model Training & Evaluation  
Feature Importance Analysis  
A Simple Web App for Predictions  

---

## Technologies Used
This project is built using the following technologies:  

- Python (Core programming language)  
- Pandas & NumPy (Data handling & processing)  
- Seaborn & Matplotlib (Data visualization)  
- Scikit-Learn (Machine Learning models)  
- Streamlit (Web-based UI for predictions)  

---

## Folder Structure
```
customer_churn/
│── data/                     # Stores datasets
│   ├── customer_churn.csv     # Raw dataset
│   ├── X_train.csv            # Processed train data
│   ├── X_test.csv             # Processed test data
│   ├── y_train.csv            # Train labels
│   ├── y_test.csv             # Test labels
│
│── models/                    # Stores trained models
│   ├── churn_model.pkl        # Final trained model
│
│── src/                       # Python scripts
│   ├── data_preprocessing.py  # Data cleaning & feature engineering
│   ├── train_model.py         # Model training
│   ├── model_evaluation.py    # Feature importance & model analysis
│   ├── predict.py             # Single customer churn prediction
│
│── app.py                     # Streamlit web app for predictions
│── requirements.txt            # Dependencies
│── README.md                   # Documentation
```

---

## Dataset Details
The dataset contains customer information and service details such as:  
- Demographics (Age, Tenure, etc.)  
- Subscription details (Contract type, Payment method, etc.)  
- Usage information (Monthly charges, Total charges, etc.)  
- Churn label (Whether the customer left or stayed)  

---

## How to Run the Project
### 1️.Install Dependencies
Make sure you have Python (3.8 or later) installed. Then, install the required libraries:  
```bash
pip install -r requirements.txt
```

### 2️.Data Preprocessing
Before training the model, we need to clean and preprocess the dataset:  
```bash
python src/data_preprocessing.py
```

### 3️.Train the Model
Once the data is prepared, train the machine learning model:  
```bash
python src/train_model.py
```
This will train the model and save it as churn_model.pkl in the `models/` folder.

### 4.Evaluate the Model
To check model performance and analyze feature importance, run:  
```bash
python src/model_evaluation.py
```

### 5.Make Predictions
To test the model on a single customer, use:  
```bash
python src/predict.py
```

### 6️.Run the Web App
Launch the Streamlit web app to make predictions using a user-friendly interface:  
```bash
streamlit run app.py
```

---

## Model Performance
- Algorithm Used: Random Forest Classifier  
- Accuracy Achieved: ~87%  
- Key Features Influencing Churn:  
  - Contract type  
  - Monthly charges  
  - Tenure  

---

## Future Improvements
- Implement XGBoost for better accuracy  
- Deploy the model using Heroku / Streamlit Cloud  
- Add SHAP analysis for better explainability  
- Integrate a real-time API for dynamic predictions  

---

## Contact & Contributions
If you have any suggestions or want to contribute, feel free to fork this project and create a pull request.  

Author: abhishek dadhel 
Email: abhiop2524@gmail.com
GitHub: https://github.com/abhiop697/ 
