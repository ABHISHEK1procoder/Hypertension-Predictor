💡 Hypertension Prediction App
This project is a machine learning-based web application developed using Streamlit that predicts whether a person is likely to have hypertension based on various health and lifestyle factors.

🔍 Objective
The main goal of this project is to provide a simple and interactive tool that helps users understand their risk of hypertension using a trained machine learning model.

⚙️ Features
User-friendly interface built with Streamlit

Accepts user input for health parameters such as:
  i) Age, Salt Intake, Stress Score
 ii) BP History, Sleep Duration, BMI
iii) Medication type, Family History
 iv) Exercise Level, Smoking Status
 
Predicts if the user is likely to have hypertension

Uses Random Forest Classifier for accurate predictions

Handles feature scaling and one-hot encoding

🧠 Machine Learning
Model: Random Forest Classifier (GridSearchCV optimized)

Preprocessing: One-hot encoding + Standard Scaler

Target: Has_Hypertension (binary classification)
