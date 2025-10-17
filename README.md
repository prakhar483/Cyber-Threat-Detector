# 🔐 Cybersecurity Intrusion Detection System using NSL-KDD Dataset (Multiclass Classification)
---

## 🧠 Project Overview

This project focuses on building a robust **Intrusion Detection System (IDS)** using **Machine Learning** and **Artificial Neural Networks (ANN)** on the widely-used **NSL-KDD Dataset**.  
The model is trained to perform **multiclass classification** — not just detecting whether traffic is normal or malicious, but **identifying the type of attack**.

🎯 **Goal:**  
Detect and classify network intrusions into one of 23 categories:  
> `Normal`, or one of 22 attack types (e.g., `neptune`, `smurf`, `satan`, `buffer_overflow`, etc.)

---

## 📁 Repository Structure
- │
- ├── Ann Project 2 Cybersecurity Intrusion Detection 15-09-2025.ipynb # Main Jupyter Notebook
- ├── CyberApp.py # Streamlit App Script
- │
- ├── KDDTrain+.txt # Training Dataset
- ├── KDDTest+.txt # Testing Dataset
- │
- ├── ann_cybersecurity_model.h5 # Trained ANN Model
- ├── scaler_cybersecurity # Feature Scaler (MinMaxScaler)
- ├── label_encoder_cyber # Label Encoder for Output
- ├── le_protocol_cyber # Encoder for 'protocol_type'
- ├── le_flag_cyber # Encoder for 'flag'
- │
- └── README.md

---

## 🧬 Dataset Summary: NSL-KDD

The NSL-KDD dataset is an improved version of the KDD’99 dataset used for evaluating network-based IDS.

- 📦 **Training Samples:** 125,973  
- 📦 **Testing Samples:** 22,544  
- 🧪 **Total Features:** 43 + 1 (Label) + 1 (Difficulty Level)  
- 🎯 **Target Classes:** 23 (1 Normal + 22 Attack Types)

---

## 🧠 Model Architecture (ANN)

A simple yet effective **Artificial Neural Network** was trained on the preprocessed features:

- **Input Layer:** 41 features (after encoding)
- **Hidden Layers:** Dense layers with ReLU activation
- **Output Layer:** Softmax activation for 23-class prediction
- **Loss Function:** Categorical Crossentropy
- **Optimizer:** Adam
- **Accuracy Achieved:** ~95% on test data

---

## 🚀 Streamlit Web App

A lightweight **Streamlit app** (`CyberApp.py`) allows you to:

✅ Upload new data  
✅ Predict the type of network connection  
✅ Display whether it's `normal` or a specific attack (e.g., `neptune`, `smurf`, etc.)
