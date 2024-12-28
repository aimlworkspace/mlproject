import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

# Streamlit App
st.title("ML Prediction App")

# Home Page
st.markdown("## Welcome to the Prediction App")
st.markdown("Use this app to predict outcomes based on your input data.")

# User Input Form
st.markdown("### Provide Your Input Data Below")

def user_input():
    gender = st.selectbox("Gender:", ["male", "female"], index=0)
    ethnicity = st.selectbox("Race/Ethnicity:", ["group A", "group B", "group C", "group D", "group E"], index=0)
    parental_level_of_education = st.selectbox(
        "Parental Level of Education:",
        ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"],
        index=0,
    )
    lunch = st.selectbox("Lunch Type:", ["standard", "free/reduced"], index=0)
    test_preparation_course = st.selectbox("Test Preparation Course:", ["none", "completed"], index=0)
    reading_score = st.number_input("Reading Score:", min_value=0.0, max_value=100.0, step=1.0)
    writing_score = st.number_input("Writing Score:", min_value=0.0, max_value=100.0, step=1.0)

    return {
        "gender": gender,
        "race_ethnicity": ethnicity,
        "parental_level_of_education": parental_level_of_education,
        "lunch": lunch,
        "test_preparation_course": test_preparation_course,
        "reading_score": reading_score,
        "writing_score": writing_score,
    }

# Input Handling
input_data = user_input()

if st.button("Predict"):
    try:
        # Convert user input into a DataFrame
        data = CustomData(
            gender=input_data["gender"],
            race_ethnicity=input_data["race_ethnicity"],
            parental_level_of_education=input_data["parental_level_of_education"],
            lunch=input_data["lunch"],
            test_preparation_course=input_data["test_preparation_course"],
            reading_score=input_data["reading_score"],
            writing_score=input_data["writing_score"],
        )

        pred_df = data.get_data_as_data_frame()

        # Instantiate and use the prediction pipeline
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        # Display the results
        st.success(f"Prediction: {results[0]}")

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")



