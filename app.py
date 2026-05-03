import streamlit as st
import pickle
import numpy as np

# --- Page Config ---
st.set_page_config(page_title="SmartCart Segmentation", page_icon="🛒", layout="centered")

# --- Custom CSS ---
st.markdown(
    """
    <style>
    .main {
        background-color: #f7f7fa;
    }
    .stButton>button {
        background: linear-gradient(90deg, #4f8cff 0%, #38e8ff 100%);
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 2em;
        margin-top: 1em;
    }
    .stTextInput>div>input, .stNumberInput>div>input {
        border-radius: 6px;
        border: 1px solid #4f8cff;
    }
    .stAlert-success {
        background-color: #e6ffed;
        color: #1a7f37;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Title & Description ---
st.markdown("<h1 style='text-align:center; color:#4f8cff;'>🛒 SmartCart Customer Segmentation</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#444;'>Predict customer cluster using Age, Income, and Spending Score.</p>", unsafe_allow_html=True)

# --- Input Section ---
st.markdown("<h3 style='color:#4f8cff;'>Enter Customer Details</h3>", unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)
with col1:
    age = st.number_input("Age", min_value=0, step=1, format="%d")
with col2:
    income = st.number_input("Annual Income", min_value=0, step=1000, format="%d")
with col3:
    spending_score = st.number_input("Spending Score", min_value=0, step=1, format="%d")

# --- Load Model & Scaler ---
try:
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler_3features.pkl", "rb"))
except Exception as e:
    st.error(f"Model or scaler not found: {e}")
    st.stop()

# --- Prediction ---
if st.button("Predict"):
    input_data = np.array([[age, income, spending_score]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)

    st.success(f"Predicted Cluster: {prediction[0]}")
    if prediction[0] == 0:
     st.success("💰 Low Value Customer")
    elif prediction[0] == 1:
     st.success("🛍️ Medium Value Customer")
    else:
     st.success("🔥 High Value Customer")