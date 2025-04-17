import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model and column structure
model = joblib.load("xgboost_model.pkl")
model_columns = joblib.load("model_columns.pkl")  # Load expected columns

st.title("Student Final Score (G3) Prediction")
st.write("Enter student details to predict the final score (G3).")

# --- You can add more fields depending on how many were used during training ---
G1 = st.number_input("G1 Score", min_value=0, max_value=20, step=1)
G2 = st.number_input("G2 Score", min_value=0, max_value=20, step=1)
school = st.selectbox("School", options=["GP", "MS"])
sex = st.selectbox("Sex", options=["F", "M"])
family_income = st.number_input("Family Income", min_value=0, step=500)

# Add more fields if your model used them...

if st.button("Predict"):
    # Create raw input data as a DataFrame
    input_data = pd.DataFrame([{
        "G1": G1,
        "G2": G2,
        "school": school,
        "sex": sex,
        "family_income": family_income
    }])

    # Add extra feature used in training
    input_data['G1_G2_avg'] = (G1 + G2) / 2

    # Match columns to model
    input_data = pd.get_dummies(input_data)

    # Reindex to match training data columns
    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    # Predict
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Final Score (G3): {round(prediction, 2)}")
