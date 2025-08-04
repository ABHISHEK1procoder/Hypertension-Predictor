import streamlit as st
import numpy as np
import pickle
import pandas as pd
import os

def load_pickle(filename):
    if not os.path.exists(filename):
        st.error(f"File '{filename}' not found in the app directory.")
        st.stop()
    with open(filename, 'rb') as file:
        return cloudpickle.load(file)

model = load_pickle('hypertension_model_new.pkl')
scalar = load_pickle('Scalar_new.pkl')

st.title("Hypertension Prediction App")

# User inputs
age = st.slider("Age", 10, 100, 30)
salt = st.selectbox("Salt Intake", ["Low", "Medium", "High"])
stress = st.slider("Stress Score", 0, 10, 5)
sleep = st.slider("Sleep Duration (hours)", 0, 12, 7)
bmi = st.slider("BMI", 10.0, 50.0, 25.0)
bp_history = st.selectbox("BP History", ["Hypertension", "Normal", "Prehypertension"])
medication = st.selectbox("Medication", ["ACE Inhibitor", "Beta Blocker", "Diuretic", "Other"])
family_history = st.selectbox("Family History of Hypertension", ["Yes", "No"])
exercise = st.selectbox("Exercise Level", ["High", "Moderate", "Low"])
smoking = st.selectbox("Smoking Status", ["Smoker", "Non-Smoker"])

# Ordinal mapping for Salt_Intake
salt_map = {"Low": 0, "Medium": 1, "High": 2}

# Initial input DataFrame
input_df = pd.DataFrame([{
    'Age': age,
    'Salt_Intake': salt_map[salt],
    'Stress_Score': stress,
    'Sleep_Duration': sleep,
    'BMI': bmi,
    'BP_History': bp_history,
    'Medication': medication,
    'Family_History': family_history,
    'Exercise_Level': exercise,
    'Smoking_Status': smoking
}])

# One-hot encode only the categorical columns that require it
categorical_cols = ['BP_History', 'Medication', 'Family_History', 'Exercise_Level', 'Smoking_Status']
input_encoded = pd.get_dummies(input_df, columns=categorical_cols)

# Ensure all model columns exist, add missing as 0
feature_columns = [
    'Age','Salt_Intake','Stress_Score','Sleep_Duration','BMI',
    'BP_History_Hypertension','BP_History_Normal','BP_History_Prehypertension',
    'Medication_ACE Inhibitor','Medication_Beta Blocker','Medication_Diuretic','Medication_Other',
    'Family_History_No','Family_History_Yes',
    'Exercise_Level_High','Exercise_Level_Low','Exercise_Level_Moderate',
    'Smoking_Status_Non-Smoker','Smoking_Status_Smoker'
]

for col in feature_columns:
    if col not in input_encoded.columns:
        input_encoded[col] = 0
input_encoded = input_encoded[feature_columns]

# Scaling (entire DataFrame as in training)
try:
    scaled_input = scalar.transform(input_encoded)
except Exception as e:
    st.error(f"Error during input scaling: {e}")
    st.write("Input to scaler:")
    st.write(input_encoded)
    st.stop()

if st.button("Predict"):
    try:
        result = model.predict(scaled_input)
        pred = result[0]
        if isinstance(pred, np.ndarray):
            pred = pred.item()
        if pred == "Yes" or pred == 1:
            st.error("⚠️ The person is likely to have Hypertension.")
        else:
            st.success("✅ The person is unlikely to have Hypertension.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")

        
