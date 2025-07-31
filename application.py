import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from src.Pipeline.predict_pipeline import CustomData,predict_pipeline
from src.utils import load_file
from src.logger import logging
from src.Exception import CustomException
import sys
import os

st.title("Student Performance Prediction App")
st.write("This app predicts student performance based on various features.")

try:

    gender = st.selectbox("Gender", ["male", "female"])
    race_ethnicity = st.selectbox("Race/Ethnicity", ["group A", "group B", "group C", "group D", "group E"])
    parental_level_of_education = st.selectbox("Parental Level of Education", ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"])
    lunch = st.selectbox("Lunch", ["standard", "free/reduced"])
    test_preparation_course = st.selectbox("Test Preparation Course", ["completed", "none"])
    reading_score = st.number_input("Reading Score", 0, 100)
    writing_score = st.number_input("Writing Score", 0, 100)
    
    if st.button("Predict"):
        st.write("Processing your input...")
        data = CustomData(
            gender=gender,
            race_ethnicity=race_ethnicity,
            parental_level_of_education=parental_level_of_education,
            lunch=lunch,
            test_preparation_course=test_preparation_course,
            reading_score=reading_score,
            writing_score=writing_score
        )
        features = data.get_data_as_dataframe()
        predict_pipe = predict_pipeline()
        prediction = predict_pipe.predict(features)
        st.subheader("Prediction Result")
        st.write(f"Predicted Score: {prediction[0]}")
        st.write("Thank you for using the Student Performance Prediction App!")
except Exception as e:
    logging.error(f"An error occurred: {e}")
    st.error(f"An error occurred: {e}")
    raise CustomException(e, sys)