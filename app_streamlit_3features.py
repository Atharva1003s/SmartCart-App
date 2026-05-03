import streamlit as st
import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler

# Title
st.title("🛒 SmartCart Simple Clustering App (3 Features)")

st.write("Predict customer cluster using Age, Income, and Spending Score.")

# Input section
st.header("Enter Customer Details")
age = st.number_input("Age", min_value=0)
income = st.number_input("Annual Income")
spending_score = st.number_input("Spending Score")

# Load model and scaler
try:
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler_3features.pkl", "rb"))
except Exception as e:
    st.error(f"Model or scaler not found: {e}")
    st.stop()

# Prediction
if st.button("Predict"):
    input_data = np.array([[age, income, spending_score]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    st.success(f"Predicted Cluster: {prediction[0]}")
