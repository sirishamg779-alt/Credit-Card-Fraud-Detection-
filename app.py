import streamlit as st
import tensorflow as tf
import joblib
import numpy as np
import pandas as pd
import os
# Suppress TensorFlow logs for a cleaner terminal
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 1. Load the saved model and scaler
@st.cache_resource
def load_assets():
    # These files must be in the same folder as app.py
    model = tf.keras.models.load_model('fraud_model.keras')
    scaler = joblib.load('scaler.pkl')
    return model, scaler

# Initialize assets
try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

# 2. UI Configuration
st.set_page_config(page_title="Fraud Guard AI", page_icon="🛡️", layout="wide")
st.title("🛡️ Credit Card Fraud Detection System")
st.write("Analyze transactions for potential fraud using Deep Learning.")

st.divider()

# 3. OPTION 1: BULK DETECTION & PERFORMANCE METRICS
st.header("📂 Bulk Detection & Model Analytics")
st.write("Upload a CSV with 30 features (Time, V1-V28, Amount). Include 'Actual_Class' to see Accuracy, Precision, and Recall.")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # Define required feature columns
    feature_cols = ['Time'] + [f'V{i}' for i in range(1, 29)] + ['Amount']
    
    # Check if features exist in the uploaded file
    if all(col in df.columns for col in feature_cols):
        with st.spinner('Calculating predictions...'):
            # Data Preprocessing
            input_data = df[feature_cols]
            scaled_data = scaler.transform(input_data)
            
            # Model Prediction
            predictions_prob = model.predict(scaled_data)
            predicted_classes = (predictions_prob >= 0.5).astype(int).flatten()
            
            # Add Results to DataFrame
            df['Fraud Probability'] = predictions_prob
            df['Status'] = ['🚨 FRAUD' if p == 1 else '✅ LEGIT' for p in predicted_classes]

        # --- CALCULATE METRICS IF LABELS ARE PROVIDED ---
        if 'Actual_Class' in df.columns:
            actual = df['Actual_Class'].values
            
            # Statistical components
            tp = np.sum((predicted_classes == 1) & (actual == 1))
            tn = np.sum((predicted_classes == 0) & (actual == 0))
            fp = np.sum((predicted_classes == 1) & (actual == 0))
            fn = np.sum((predicted_classes == 0) & (actual == 1))

            # Metric Formulas
            accuracy = (tp + tn) / len(actual)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            # Display metrics in 3 columns
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy Score", f"{accuracy:.2%}")
            col2.metric("Precision", f"{precision:.2%}")
            col3.metric("Recall (Detection Rate)", f"{recall:.2%}")
            st.divider()

        st.success("Analysis Complete!")
        st.dataframe(df.style.background_gradient(subset=['Fraud Probability'], cmap='Reds'))
    else:
        st.error("CSV Error: Missing required columns. Ensure Time, V1-V28, and Amount are present.")

st.divider()

# 4. OPTION 2: MANUAL TRANSACTION ENTRY (SIDEBAR)
st.sidebar.header("📝 Single Check")
with st.sidebar:
    st.write("Enter details manually:")
    in_time = st.number_input("Time", value=0.0)
    in_amount = st.number_input("Amount", value=0.0)
    
    with st.expander("V1 - V28 Features"):
        v_inputs = []
        for i in range(1, 29):
            v_val = st.number_input(f"V{i}", value=0.0, key=f"manual_v{i}")
            v_inputs.append(v_val)

if st.sidebar.button("Predict Single Transaction"):
    # Reshape and Scale
    features = np.array([in_time] + v_inputs + [in_amount]).reshape(1, -1)
    scaled_features = scaler.transform(features)
    
    # Predict
    prob = model.predict(scaled_features)[0][0]
    
    if prob >= 0.5:
        st.sidebar.error(f"🚨 FRAUD DETECTED\nProbability: {prob:.4f}")
    else:
        st.sidebar.success(f"✅ TRANSACTION LEGIT\nProbability: {prob:.4f}")