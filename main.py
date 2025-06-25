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
st.title("📊 Segmentasi Customer Mall Menggunakan K-Means")

st.markdown(
    "Masukkan data pelanggan untuk mengetahui klaster "
    "dan rekomendasi strategi pemasaran."
)

# ----------------------- FORM INPUT ------------------------
with st.form("input_form"):
    gender = st.selectbox("Gender", ["Male", "Female"])
    age    = st.slider("Age", 15, 70, 30)
    income = st.slider("Annual Income (k$)", 10, 150, 50)
    score  = st.slider("Spending Score (1-100)", 1, 100, 50)
    submitted = st.form_submit_button("Prediksi Klaster")

# ----------------------- PREDIKSI & OUTPUT ------------------------
if submitted:
    gender_encoded = 1 if gender == "Female" else 0
    input_data  = np.array([[gender_encoded, age, income, score]])
    input_scaled = scaler.transform(input_data)

    cluster_idx = int(kmeans_model.predict(input_scaled)[0])     # 0..9
    cluster_num = cluster_idx + 1                                # 1..10

    # ---- Segmentasi bisnis --------------------------------------
    if 1 <= cluster_num <= 4:
        segment = "Rendah"
        desc = (
            "Kelompok ini terdiri dari pelanggan yang **belum terlalu aktif berbelanja** "
            "atau hanya sesekali datang. Fokusnya adalah _engagement_: "
            "program edukasi, kupon diskon, atau promo menarik agar naik ke level Premium."
        )
    elif 5 <= cluster_num <= 6:
        segment = "Menengah"
        desc = (
            "Ini adalah **pelanggan potensial untuk naik kelas**. "
            "Mereka sudah cukup sering belanja dan berkontribusi penting. "
            "Motivasi mereka dengan poin reward, diskon loyalitas, atau program cashback."
        )
    else:  # 7-9 (dan 10 jika ada)
        segment = "Tinggi"
        desc = (
            "Pelanggan ini adalah yang **paling bernilai** bagi mall. "
            "Mereka belanja banyak dan sering. Berikan layanan istimewa seperti lounge VIP, "
            "penawaran eksklusif, undangan acara spesial, atau parkir khusus."
        )

    # ----------------------- DISPLAY -----------------------------
    st.success(f"✅ Klaster prediksi: **Cluster {cluster_num}** → Segmen **{segment}**")
    st.markdown(desc)

    st.markdown("---")
    st.subheader("🔍 Alasan Masuk Klaster Ini:")
    center = kmeans_model.cluster_centers_[cluster_idx]
    for i, feat in enumerate(features):
        diff = abs(input_scaled[0][i] - center[i])
        if diff < 0.5:
            st.markdown(f"- ✔️ **{feat}** mirip rata-rata klaster.")
        else:
            st.markdown(f"- ℹ️ **{feat}** cukup berbeda, tapi masih paling dekat ke klaster ini.")

    st.markdown("---")
    st.info("Model menggunakan 4 fitur: Gender, Age, Income, Spending Score")
