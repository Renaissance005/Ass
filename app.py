import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load model and expected column names
model = joblib.load("xgboost_model.pkl")
model_columns = joblib.load("model_columns.pkl")

st.set_page_config(page_title="G3 Prediction", layout="centered")
st.title("🎓 Predict Student's Final Score (G3)")
st.write("Please enter the student's scores and background details:")

# Input form (same as your HTML)
with st.form("predict_form"):
    G1 = st.number_input("G1 Score", min_value=0, max_value=20, step=1)
    G2 = st.number_input("G2 Score", min_value=0, max_value=20, step=1)
    family_income = st.number_input("Family Income", min_value=0, step=100)

    # You can uncomment and use these if your model used them
    # school = st.selectbox("School", ["GP", "MS"])
    # sex = st.selectbox("Sex", ["F", "M"])

    submitted = st.form_submit_button("Predict")

if submitted:
    # Create DataFrame from input
    input_data = pd.DataFrame([{
        "G1": G1,
        "G2": G2,
        "family_income": family_income,
        # "school": school,
        # "sex": sex
    }])

    # Create average score feature if used during training
    input_data['G1_G2_avg'] = (G1 + G2) / 2

    # Handle categorical if used (uncomment if needed)
    input_data = pd.get_dummies(input_data)
    input_data = input_data.reindex(columns=model_columns, fill_value=0)

    # Make prediction
    prediction = model.predict(input_data)[0]
    st.success(f"📊 Predicted Final Score (G3): **{round(prediction, 2)}** out of 20")
