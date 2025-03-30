from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load("xgboost_model.pkl")

# Initialize Flask app
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Get data from request
    df = pd.DataFrame([data])  # Convert to DataFrame
    
    # Ensure the feature order is correct
    df['G1_G2_avg'] = (df['G1'] + df['G2']) / 2

    # Make prediction
    prediction = model.predict(df)[0]
    
    return jsonify({"predicted_G3": round(prediction, 2)})

if __name__ == '__main__':
    app.run(debug=True)
