import streamlit as st
import pandas as pd
import numpy as np
import pickle
from pathlib import Path

# --------------------------------------------------------------------
# Path setup supaya aman (lokal & cloud)
# --------------------------------------------------------------------
APP_DIR = Path(__file__).parent
MODEL_PATH = APP_DIR / "final_model.pkl"
IMG_PATH = APP_DIR / "ecommercepict.png"

# --------------------------------------------------------------------
# Load model
# --------------------------------------------------------------------
@st.cache_resource
def load_model(pickle_path=MODEL_PATH):
    try:
        with open(pickle_path, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        st.error("Model file not found. Please check deployment folder.")
        return None

model = load_model()

# --------------------------------------------------------------------
# Streamlit UI
# --------------------------------------------------------------------
st.set_page_config(page_title="E-Commerce Churn Prediction", layout="wide")

st.title("E-Commerce Customer Churn Prediction")

# tampilkan gambar
if IMG_PATH.exists():
    st.image(str(IMG_PATH), use_container_width=True)
else:
    st.warning("Gambar ecommercepict.png tidak ditemukan.")

st.markdown("""
Aplikasi ini digunakan untuk memprediksi **churn pelanggan e-commerce** berdasarkan fitur-fitur customer.  
Silakan isi form di bawah untuk mencoba prediksi.
""")

# --------------------------------------------------------------------
# Input form
# --------------------------------------------------------------------
with st.form("prediction_form"):
    st.subheader("Masukkan Data Customer:")

    Tenure = st.number_input("Tenure (bulan)", min_value=0, max_value=100, value=12)
    WarehouseToHome = st.number_input("Warehouse to Home (km)", min_value=0, max_value=100, value=10)
    HourSpendOnApp = st.number_input("Hour Spend on App", min_value=0, max_value=24, value=2)
    NumberOfDeviceRegistered = st.number_input("Number of Device Registered", min_value=1, max_value=10, value=1)
    SatisfactionScore = st.slider("Satisfaction Score", min_value=1, max_value=5, value=3)
    NumberOfAddress = st.number_input("Number of Address", min_value=1, max_value=10, value=1)
    Complain = st.selectbox("Complain", [0, 1])
    OrderAmountHikeFromlastYear = st.number_input("Order Amount Hike From Last Year (%)", min_value=0, max_value=100, value=10)
    CouponUsed = st.number_input("Coupon Used", min_value=0, max_value=20, value=1)
    OrderCount = st.number_input("Order Count", min_value=0, max_value=100, value=5)
    DaySinceLastOrder = st.number_input("Day Since Last Order", min_value=0, max_value=100, value=10)
    CashbackAmount = st.number_input("Cashback Amount", min_value=0, max_value=1000, value=50)

    submitted = st.form_submit_button("Predict Churn")

# --------------------------------------------------------------------
# Prediction
# --------------------------------------------------------------------
if submitted:
    if model is None:
        st.error("Model tidak tersedia, prediksi tidak bisa dilakukan.")
    else:
        input_data = np.array([[Tenure, WarehouseToHome, HourSpendOnApp,
                                NumberOfDeviceRegistered, SatisfactionScore,
                                NumberOfAddress, Complain,
                                OrderAmountHikeFromlastYear, CouponUsed,
                                OrderCount, DaySinceLastOrder,
                                CashbackAmount]])
        
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

        st.subheader("Hasil Prediksi")
        if prediction == 1:
            st.error(f"Customer diprediksi **CHURN** dengan probabilitas {probability:.2%}")
        else:
            st.success(f"Customer diprediksi **TIDAK CHURN** dengan probabilitas {1-probability:.2%}")
