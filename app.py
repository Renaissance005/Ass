import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model
model = joblib.load("xgboost_model.pkl")

# Streamlit UI
st.title("Student Final Score (G3) Prediction")
st.write("Enter student details to predict the final score (G3).")

# User input fields
G1 = st.number_input("G1 Score", min_value=0, max_value=20, step=1)
G2 = st.number_input("G2 Score", min_value=0, max_value=20, step=1)
family_income = st.number_input("Family Income", min_value=0, step=500)

# Prediction function
if st.button("Predict"):
    input_data = pd.DataFrame([[G1, G2, family_income]], columns=['G1', 'G2', 'family_income'])
    input_data['G1_G2_avg'] = (input_data['G1'] + input_data['G2']) / 2  # Add feature used in training

    prediction = model.predict(input_data)[0]
    
    st.success(f"Predicted Final Score (G3): {round(prediction, 2)}")
