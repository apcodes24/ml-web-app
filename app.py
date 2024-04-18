import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open('lr_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Function to preprocess input data
def preprocess_input(age, sex, bmi, children, smoker, region):
    # Encode categorical variables
    sex_encoded = 1 if sex == 'female' else 0
    smoker_encoded = 1 if smoker == 'no' else 0
    region_encoded = {'southeast': 0, 'southwest': 1, 'northeast': 2, 'northwest': 3}[region]
    # Return preprocessed input as numpy array
    return np.array([[age, sex_encoded, bmi, children, smoker_encoded, region_encoded]])

# Function to predict charges
def predict_charges(age, sex, bmi, children, smoker, region):
    input_data = preprocess_input(age, sex, bmi, children, smoker, region)
    prediction = model.predict(input_data)[0]
    return prediction

# Streamlit app
def main():
    st.title('Medical Insurance Cost Prediction')
    
    # Input fields
    age = st.slider('Age', min_value=18, max_value=100, value=25, step=1)
    sex = st.selectbox('Sex', ['male', 'female'])
    bmi = st.number_input('BMI', min_value=10.0, max_value=50.0, value=25.0, step=0.1)
    children = st.number_input('Children', min_value=0, max_value=10, value=0, step=1)
    smoker = st.selectbox('Smoker', ['yes', 'no'])
    region = st.selectbox('Region', ['southeast', 'southwest', 'northeast', 'northwest'])
    
    # Predict button
    if st.button('Predict Charges'):
        prediction = predict_charges(age, sex, bmi, children, smoker, region)
        st.success(f'Predicted cost: ${prediction:.2f}')

if __name__ == '__main__':
    main()
