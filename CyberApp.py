import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# --- Attractive Project Description ---
st.set_page_config(page_title="Cyber Threat Classifier", page_icon="🛡️", layout="centered")
st.markdown("""
# 🛡️ Cyber Threat Detection with ANN  
Detect malicious network activity and classify cyber threats using Artificial Neural Networks and the NSL-KDD dataset.

---

### 🚀 About This App
This app predicts the type of cyber threat (or normal traffic) based on network connection features.  
It leverages a trained ANN model on the **NSL-KDD** dataset, a benchmark for intrusion detection research.

- **Data Source:** [NSL-KDD on Kaggle](https://www.kaggle.com/datasets/hassan06/nslkdd)  
- **Goal:** Classify connections as one of:  
  - 🟢 Normal  
  - 🔴 DoS (Denial of Service)  
  - 🟡 Probe (Scanning)  
  - 🟠 R2L (Remote to Local)  
  - 🟣 U2R (User to Root)  
  - ⚪ Other

---

### 📊 How It Works
1. **Input**: Enter network connection details below.
2. **Processing**: Features are encoded & scaled as per model training.
3. **Prediction**: The ANN model predicts the cyber threat category.

---
""")

# --- Load model and encoders ---
@st.cache_resource
def load_all():
    model = load_model('ann_cybersecurity_model.h5')
    le_flag = joblib.load('le_flag_cyber')
    le_protocol = joblib.load('le_protocol_cyber')
    scaler = joblib.load('scaler_cybersecurity')
    label_encoder = joblib.load('label_encoder_cyber')
    return model, le_flag, le_protocol, scaler, label_encoder

model, le_flag, le_protocol, scaler, label_encoder = load_all()

# --- Streamlit UI ---
st.header("🔍 Predict Cyber Threat Type")

protocol_types = ['tcp', 'udp', 'icmp']
flags = ['SF', 'S0', 'REJ', 'RSTR', 'SH', 'RSTO', 'S1', 'RSTOS0', 'S3', 'S2', 'OTH']

with st.form("prediction_form"):
    col1, col2 = st.columns(2)
    with col1:
        protocol_type = st.selectbox("🌐 Protocol Type", protocol_types)
        flag = st.selectbox("🚩 Flag", flags)
        dst_host_srv_serror_rate = st.number_input("🔸 dst_host_srv_serror_rate", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
        srv_serror_rate = st.number_input("🔸 srv_serror_rate", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
        dst_host_serror_rate = st.number_input("🔸 dst_host_serror_rate", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
        serror_rate = st.number_input("🔸 serror_rate", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    with col2:
        srv_count = st.number_input("🔹 srv_count", min_value=0, max_value=100, value=1, step=1)
        wrong_fragment = st.number_input("🔹 wrong_fragment", min_value=0, max_value=10, value=0, step=1)
        dst_host_diff_srv_rate = st.number_input("🔹 dst_host_diff_srv_rate", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
        same_srv_rate = st.number_input("🔹 same_srv_rate", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
        dst_host_srv_diff_host_rate = st.number_input("🔹 dst_host_srv_diff_host_rate", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
        diff_srv_rate = st.number_input("🔹 diff_srv_rate", min_value=0.0, max_value=1.0, value=0.0, step=0.01)
    submitted = st.form_submit_button("🔎 Predict")

if submitted:
    # Prepare input
    input_data = [
        le_protocol.transform([protocol_type])[0],
        le_flag.transform([flag])[0],
        dst_host_srv_serror_rate,
        srv_serror_rate,
        dst_host_serror_rate,
        serror_rate,
        srv_count,
        wrong_fragment,
        dst_host_diff_srv_rate,
        same_srv_rate,
        dst_host_srv_diff_host_rate,
        diff_srv_rate
    ]
    input_array = np.array(input_data).reshape(1, -1)
    input_scaled = scaler.transform(input_array)

    # Predict
    pred = model.predict(input_scaled)
    pred_class = np.argmax(pred, axis=1)
    pred_label = label_encoder.inverse_transform(pred_class)[0]
    confidence = np.max(pred) * 100

    color_map = {
        "Normal": "🟢",
        "DoS": "🔴",
        "Probe": "🟡",
        "R2L": "🟠",
        "U2R": "🟣",
        "Other": "⚪"
    }
    emoji = color_map.get(pred_label, "❓")

    st.success(f"## {emoji} Predicted Cyber Threat Type: **{pred_label}**")
    st.markdown(
        f"<div style='font-size:18px;'>Prediction Confidence: <b>{confidence:.2f}%</b></div>",
        unsafe_allow_html=True
    )

st.markdown("---")
st.caption("Made with ❤️ by Prakhar Dwivedi using Streamlit & Keras | Data: NSL-KDD")