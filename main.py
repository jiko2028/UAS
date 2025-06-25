import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ----------------------- LOAD MODEL & SCALER ------------------------
@st.cache_resource
def load_model_scaler():
    with open("kmeans_model_4f.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler_4f.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

kmeans_model, scaler = load_model_scaler()
features = ['GenderEncoded', 'Age', 'Annual Income (k$)', 'Spending Score (1-100)']

# ----------------------- UI ------------------------
st.set_page_config(page_title="Customer Segmentation", layout="centered")
st.title("📊 Customer Segmentation using K-Means")

st.markdown("Masukkan data pelanggan untuk mengetahui termasuk klaster yang mana berdasarkan **KMeans Clustering**:")

# Input Form
with st.form("input_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.slider("Age", 15, 70, 30)
    income = st.slider("Annual Income (k$)", 10, 150, 50)
    score = st.slider("Spending Score (1-100)", 1, 100, 50)
    submitted = st.form_submit_button("Prediksi Klaster")

if submitted:
    gender_encoded = 1 if gender == "Female" else 0
    input_data = np.array([[gender_encoded, age, income, score]])
    input_scaled = scaler.transform(input_data)

    cluster = kmeans_model.predict(input_scaled)[0]
    center = kmeans_model.cluster_centers_[cluster]

    st.success(f"✅ Pelanggan ini masuk **Cluster {cluster}**")
    st.markdown("---")
    st.subheader("🔍 Penjelasan Kenapa Masuk ke Klaster Ini:")
    for i, feat in enumerate(features):
        diff = abs(input_scaled[0][i] - center[i])
        if diff < 0.5:
            st.markdown(f"- ✔️ **{feat}** mirip dengan nilai rata-rata klaster.")
        else:
            st.markdown(f"- ℹ️ **{feat}** cukup berbeda tapi tetap paling mendekati klaster ini.")
    
    st.markdown("---")
    st.info("Model ini menggunakan 4 fitur: Gender, Age, Income, Spending Score")
