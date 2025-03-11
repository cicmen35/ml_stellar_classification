import streamlit as st
import joblib
import numpy as np

# load model and encoder
model = joblib.load("stellar_model.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# streamlit ui
st.title("Stellar Classification Predictor")

st.write("Enter the stellar parameters to predict its class.")

# Define user input fields based on dataset features
feature_columns = [
    "alpha", "delta", "u", "g", "r", "i", "z", "redshift",
    "run_ID", "rerun_ID", "cam_col", "field_ID",
    "spec_obj_ID", "plate", "MJD", "fiber_ID"
]

inputs = {}
for col in feature_columns:
    inputs[col] = st.number_input(f"{col} Value", min_value=0.0, max_value=100000.0, value=10.0)

# Prediction button
if st.button("PREDICT"):
    input_data = np.array([[inputs[col] for col in feature_columns]])
    prediction = model.predict(input_data)
    predicted_class = label_encoder.inverse_transform(prediction)[0]

    st.success(f"Predicted stellar class: **{predicted_class}**")
