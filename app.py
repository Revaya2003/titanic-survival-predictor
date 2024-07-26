import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Set the style of seaborn
sns.set(style="whitegrid")

# Load model dan scaler yang sudah dilatih
model = joblib.load('titanic_predict.sav')
scaler = joblib.load('scaler.pkl')

# Halaman utama aplikasi
st.set_page_config(page_title='Titanic Survival Predictor', layout='wide')
st.title('Titanic Survival Predictor')
st.write("""
    Selamat datang di aplikasi **Titanic Survival Predictor**! 
    Masukkan informasi penumpang di bawah ini untuk memprediksi apakah penumpang kemungkinan besar selamat atau tidak.
""")

# Sidebar untuk input data
st.sidebar.header('Input Data')
pclass = st.sidebar.selectbox('Class', [1, 2, 3], index=0, format_func=lambda x: f"Class {x}")
age = st.sidebar.slider('Age', min_value=0, max_value=100, value=30, step=1)
fare = st.sidebar.slider('Fare', min_value=0.0, max_value=512.3292, value=70.0, step=0.1)
sex = st.sidebar.selectbox('Sex', ['Male (0)', 'Female (1)'], index=0)
sex = 0 if sex == 'Male (0)' else 1
embarked = st.sidebar.selectbox('Embarked', [0, 1, 2], index=0)  # Use integer directly

# Display input data
st.write("### Data Input")
st.write(f"Pclass: {pclass}")
st.write(f"Age: {age}")
st.write(f"Fare: {fare}")
st.write(f"Sex: {'Male' if sex == 0 else 'Female'}")
st.write(f"Embarked: {'Cherbourg' if embarked == 0 else 'Queenstown' if embarked == 1 else 'Southampton'}")

# Membuat tombol untuk prediksi
if st.sidebar.button('Prediksi'):
    try:
        # Konversi input menjadi array numpy
        inputs = np.array([[pclass, age, fare, sex, embarked]])

        # Normalisasi input data
        inputs_scaled = scaler.transform(inputs)

        # Lakukan prediksi
        prediction = model.predict(inputs_scaled)

        # Tampilkan hasil prediksi dengan styling khusus
        st.write("### Hasil Prediksi")
        if prediction[0] == 1:
            st.markdown('<div style="background-color: #28a745; color: white; padding: 10px; border-radius: 5px;">Selamat! Penumpang ini kemungkinan besar selamat.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div style="background-color: #dc3545; color: white; padding: 10px; border-radius: 5px;">Tidak Selamat! Penumpang ini kemungkinan besar tidak selamat.</div>', unsafe_allow_html=True)
        
        # Visualisasi Input Data
        fig, ax = plt.subplots(figsize=(8, 4))
        sns.barplot(x=['Pclass', 'Age', 'Fare', 'Sex', 'Embarked'], y=inputs.flatten(), palette='viridis', ax=ax)
        ax.set_title('Distribusi Fitur Input')
        ax.set_ylabel('Nilai')
        ax.set_xlabel('Fitur')
        st.pyplot(fig)

        # Statistik Model
        st.write("### Statistik Model")
        st.write(f"Model yang digunakan: Logistic Regression")
        st.write(f"Jumlah data pelatihan: {scaler.mean_.size} fitur")
        
    except Exception as e:
        st.error(f"Terjadi kesalahan: {e}")

# Footer
st.markdown("""
    ---
    **Aplikasi ini menggunakan model machine learning untuk memprediksi kelangsungan hidup penumpang Titanic.**
    - **Model**: Logistic Regression
    - **Sumber Data**: Titanic - Machine Learning from Disaster Dataset
    - **Pengembang**: Revaya Rizqia Pasya
    - **Versi**: 1.0
    - **Kontak**: revayarizqia@gmmail.com
""")
