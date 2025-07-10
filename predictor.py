import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Load the trained model
model = tf.keras.models.load_model('model.h5')

# Load the scaler

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


## streamlit app
st.title('DIABETES RISK PREDICTOR')

# User input
pregnancies = st.number_input('Enter number of pregnancies(if any)')
Glucose = st.number_input('Enter your glucose level' , max_value=400.0)
BloodPressure = st.number_input('Enter Blood Pressure Level:', max_value=200.0)
SkinThickness = st.number_input('Triceps skinfold thickness (fat under skin) in mm:', max_value=100.0)
Insulin = st.number_input("Enter your insulin level (μU/mL), if known", min_value=0.0, step=0.1 , max_value=900.0)
BMI = st.number_input('Enter your BMI', max_value=70.0)
Age = st.slider('Enter your age', min_value=1.0, max_value=150.0 ,value=10.0)
family_history = st.selectbox("Do you have a family history of diabetes?", ["No", "Yes"])
family_history_binary = 1 if family_history == "Yes" else 0
# Prepare the input data
input_data = pd.DataFrame({
 'Pregnancies':[pregnancies],
 'Glucose': [Glucose],
 'BloodPressure': [BloodPressure],
 'SkinThickness': [SkinThickness],
 'Insulin': [Insulin],
 'BMI': [BMI],
 'Age': [Age],
 'FamilyHistory': [family_history_binary]
})


# Scale the input data
input_data_scaled = scaler.transform(input_data)


# Predict risk
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

st.write(f'Risk of diabetes Probability: {prediction_proba:.2f}')

if prediction_proba > 0.5:
    st.write('The customer is likely to develop diabetes.')
else:
    st.write('The customer is not likely to develop diabetes.')
