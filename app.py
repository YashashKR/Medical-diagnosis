import streamlit as st
import joblib
import numpy as np

# Load model
model = joblib.load('model.pkl')

# Title
st.title('Medical Diagnosis Prediction')

# User input
st.sidebar.header('Patient Test Results')
blood_pressure = st.sidebar.slider('Blood Pressure (mmHg)', 80, 180, 120)
cholesterol = st.sidebar.slider('Cholesterol Level (mg/dL)', 100, 300, 200)

input_features = np.array([[blood_pressure, cholesterol]])

# Prediction
if st.sidebar.button('Diagnose'):
    prediction = model.predict(input_features)
    probability = model.predict_proba(input_features)

    result = "Disease Present" if prediction[0] == 1 else "No Disease"
    st.write(f'### Prediction: {result}')
    st.write(f'Probability of No Disease: {probability[0][0]:.2f}')
    st.write(f'Probability of Disease: {probability[0][1]:.2f}')
