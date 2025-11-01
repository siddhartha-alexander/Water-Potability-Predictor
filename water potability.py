import streamlit as st
import numpy as np
import pandas as pd
import joblib

# ==================== Page Styling ====================
st.set_page_config(page_title="Water Potability Predictor", page_icon="💧", layout="centered")

st.markdown("""
    <style>
        .stApp {
            background-color: #f2f8fc;
            font-family: 'Segoe UI';
        }
        h1, h2, h3 {
            color: #0c4a6e;
        }
        .stButton>button {
            background-color: #0ea5e9;
            color: white;
            border-radius: 8px;
            height: 3em;
            width: 100%;
        }
        .stButton>button:hover {
            background-color: #0284c7;
        }
    </style>
""", unsafe_allow_html=True)

# ==================== Load Model and Scaler ====================
model = joblib.load('model.joblib')
scaler = joblib.load('scaler.joblib')

# ==================== App Title and Description ====================
st.title("💧 Water Potability Prediction App")
st.write("This app predicts whether a given water sample is **safe for drinking** based on its chemical properties.")

# ==================== Sidebar Inputs ====================
st.sidebar.header("🔹 Enter Water Quality Parameters")

ph = st.sidebar.number_input("pH Value", min_value=0.0, max_value=14.0, value=7.0)
hardness = st.sidebar.number_input("Hardness", min_value=0.0, value=200.0)
solids = st.sidebar.number_input("Solids", min_value=0.0, value=20000.0)
chloramines = st.sidebar.number_input("Chloramines", min_value=0.0, value=7.0)
sulfate = st.sidebar.number_input("Sulfate", min_value=0.0, value=300.0)
conductivity = st.sidebar.number_input("Conductivity", min_value=0.0, value=400.0)
organic_carbon = st.sidebar.number_input("Organic Carbon", min_value=0.0, value=10.0)
trihalomethanes = st.sidebar.number_input("Trihalomethanes", min_value=0.0, value=70.0)
turbidity = st.sidebar.number_input("Turbidity", min_value=0.0, value=4.0)

# ==================== Single Prediction ====================
features = np.array([[ph, hardness, solids, chloramines, sulfate, conductivity,
                      organic_carbon, trihalomethanes, turbidity]])

scaled_features = scaler.transform(features)

st.markdown("---")
st.subheader("🔍 Predict Water Potability")

if st.button("Predict Potability"):
    prediction = model.predict(scaled_features)
    prob = model.predict_proba(scaled_features)[0][1] * 100

    if prediction[0] == 1:
        st.success(f"✅ The water is **Potable (Safe for Drinking)**.\n\n**Confidence:** {prob:.2f}%")
    else:
        st.error(f"🚫 The water is **Not Potable (Unsafe for Drinking)**.\n\n**Confidence:** {100 - prob:.2f}%")

# ==================== Batch Prediction (CSV Upload) ====================
st.markdown("---")
st.subheader("📁 Upload CSV for Batch Prediction")

st.info("Ensure your CSV has these columns in order: **ph, Hardness, Solids, Chloramines, Sulfate, Conductivity, Organic_carbon, Trihalomethanes, Turbidity**")

uploaded_file = st.file_uploader("Upload a CSV file", type=['csv'])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### Uploaded Data Preview:")
    st.dataframe(data.head())

    try:
        scaled_data = scaler.transform(data)
        predictions = model.predict(scaled_data)
        probas = model.predict_proba(scaled_data)[:, 1]
        data['Predicted Potability'] = predictions
        data['Confidence (%)'] = (probas * 100).round(2)

        st.write("### Predictions:")
        st.dataframe(data)

        csv = data.to_csv(index=False).encode('utf-8')
        st.download_button("Download Predictions as CSV", data=csv,
                           file_name="Water_Predictions.csv", mime='text/csv')
    except Exception as e:
        st.error("⚠️ Please ensure your CSV file matches the required feature order.")
        st.exception(e)

# ==================== Footer ====================
st.markdown("---")
st.caption(" | Model: Random Forest Classifier | Accuracy: 66% | ROC-AUC: 0.63")
