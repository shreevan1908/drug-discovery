import streamlit as st
import tensorflow as tf
import joblib
import numpy as np
import pandas as pd

# Load model and scaler
model = tf.keras.models.load_model("drug_model.h5")
scaler = joblib.load("scaler.pkl")
feature_names = pd.read_csv("feature_names.txt", header=None)[0].tolist()

st.title("🧪 AI Drug Activity Predictor")

# Input form for all features
input_data = {}
st.subheader("Enter Drug Descriptor Values:")

for name in feature_names:
    input_data[name] = st.number_input(f"{name}", value=0.0)

# Convert input to array
if st.button("Predict pXC50"):
    input_array = np.array([list(input_data.values())])
    if input_array.shape[1] != len(feature_names):
        st.error("Input data does not match the expected number of features.")
    else:
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0][0]
        st.success(f"Predicted pXC50 Value: {prediction:.2f}")