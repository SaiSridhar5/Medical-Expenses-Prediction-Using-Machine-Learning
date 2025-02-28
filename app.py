import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
model = joblib.load('expenses_predictor_model.pkl')
scaler = joblib.load('scaler.pkl')

# Title of the web app
st.title('Medical Expenses Prediction')

# Input fields for the user
st.header('Enter the details for prediction')

# Input form for required fields
age = st.number_input('Age', min_value=18, max_value=100, step=1)
bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, step=0.1)
children = st.number_input('Number of children', min_value=0, max_value=10, step=1)

# For smoker, let's use a dropdown (Yes = 1, No = 0)
smoker = st.selectbox('Do you smoke?', ['No', 'Yes'])
smoker_map = {'No': 0, 'Yes': 1}
smoker = smoker_map[smoker]

# **These fields are for UI only and not used in the prediction model**
# For sex, we need to use a dropdown (Male = 0, Female = 1)
sex = st.selectbox('Sex ', ['Male', 'Female'])

# For region, assuming possible values like 'North', 'South', 'East', 'West' (could be different based on dataset)
region = st.selectbox('Region ', ['North', 'South', 'East', 'West'])

# Prepare the input data for prediction
# Only using age, bmi, children, and smoker for prediction (exclude sex and region)
input_data = np.array([[age, bmi, children, smoker]])

# Scale the input data using the same scaler used for training
input_scaled = scaler.transform(input_data)

# Prediction button
if st.button('Predict'):
    # Predict using the model
    prediction = model.predict(input_scaled)
    
    # Display the result
    st.subheader(f'Predicted Medical Expenses: ₹{prediction[0]:.2f}')
