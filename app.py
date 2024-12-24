# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained model, scaler, and label encoders
model = joblib.load('stroke_model.pkl')
scaler = joblib.load('scaler.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Define the prediction function
def predict_stroke(input_data):
    # Convert input data to DataFrame
    input_df = pd.DataFrame([input_data])

    # List of categorical columns
    categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status', 'hypertension', 'heart_disease']

    # Ensure all text is lowercase
    for col in categorical_cols:
        input_df[col] = input_df[col].str.lower()

    # Encode categorical variables
    for col in label_encoders:
        le = label_encoders[col]
        input_df[col] = le.transform(input_df[col])

    # Feature scaling
    input_scaled = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)

    return prediction[0], prediction_proba[0][1]  # Return class and probability for stroke

# Streamlit app
def main():
    st.title("Stroke Detection Prediction App")
    st.write("""
    Provide the following details to predict the likelihood of a stroke.
    """)

    # Input fields

    # Gender
    gender_options = {'Male': 'male', 'Female': 'female'}
    gender_display = st.selectbox('Gender', list(gender_options.keys()))
    gender = gender_options[gender_display]

    # Age
    age = st.slider('Age', 1, 100, 50)

    # Hypertension
    hypertension_options = {'No': '0', 'Yes': '1'}
    hypertension_display = st.selectbox('Hypertension', list(hypertension_options.keys()))
    hypertension = hypertension_options[hypertension_display]

    # Heart Disease
    heart_disease_options = {'No': '0', 'Yes': '1'}
    heart_disease_display = st.selectbox('Heart Disease', list(heart_disease_options.keys()))
    heart_disease = heart_disease_options[heart_disease_display]

    # Ever Married
    ever_married_options = {'Yes': 'yes', 'No': 'no'}
    ever_married_display = st.selectbox('Ever Married', list(ever_married_options.keys()))
    ever_married = ever_married_options[ever_married_display]

    # Work Type
    work_type_options = {
        'Private': 'private',
        'Self-employed': 'self-employed',
        'Govt_job': 'govt_job',
        'Children': 'children',
        'Never_worked': 'never_worked'
    }
    work_type_display = st.selectbox('Work Type', list(work_type_options.keys()))
    work_type = work_type_options[work_type_display]

    # Residence Type
    Residence_type_options = {'Urban': 'urban', 'Rural': 'rural'}
    Residence_type_display = st.selectbox('Residence Type', list(Residence_type_options.keys()))
    Residence_type = Residence_type_options[Residence_type_display]

    # Average Glucose Level
    avg_glucose_level = st.slider('Average Glucose Level', 50.0, 300.0, 100.0)

    # BMI
    bmi = st.slider('BMI', 10.0, 60.0, 20.0)

    # Smoking Status
    smoking_status_options = {
        'Never smoked': 'never smoked',
        'Unknown': 'unknown',
        'Formerly smoked': 'formerly smoked',
        'Smokes': 'smokes'
    }
    smoking_status_display = st.selectbox('Smoking Status', list(smoking_status_options.keys()))
    smoking_status = smoking_status_options[smoking_status_display]

    # Prediction
    if st.button('Predict'):
        input_data = {
            'gender': gender,
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'ever_married': ever_married,
            'work_type': work_type,
            'Residence_type': Residence_type,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'smoking_status': smoking_status
        }

        prediction, probability = predict_stroke(input_data)

        if prediction == 1:
            st.error(f'The model predicts a high risk of stroke with a probability of {probability:.2f}.')
        else:
            st.success(f'The model predicts a low risk of stroke with a probability of {probability:.2f}.')

        # Optionally, display additional information
        st.write('**Prediction Details:**')
        st.write(f'Probability of Stroke: {probability:.2f}')
        st.write(f'Probability of No Stroke: {1 - probability:.2f}')

if __name__ == '__main__':
    main()