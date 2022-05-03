# -*- coding: utf-8 -*-
"""
Created on Tue May  3 16:15:35 2022

@author: vishal
"""

import numpy as np
import pickle 
import streamlit as st



# loading the saved model
loaded_model = pickle.load(open("D:/ProgramData/ML_Siddhardhan/1. Project Diabetes/diabetes_trained_model.sav", "rb"))



# creating a function for Prediction

def diabetes_prediction(input_data):
    
    input_data_as_np_array = np.asarray(input_data)

    input_data_reshaped = input_data_as_np_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)

    if (prediction[0]==0):
      return "The person is not diabetic."
  
    return "The person is diabetic."
  
    

def main():
    
    # giving a title
    st.title("Diabetes Prediction by Vishal Jaiswar")
    
    
    # getting input data from the user
    Pregnancies = st.text_input("Number of pregnancies")
    Glucose = st.text_input("Glucose level")
    BloodPressure = st.text_input("Blood pressure")
    SkinThickness = st.text_input("Skin thickness value")
    Insulin = st.text_input("Insulin level")
    BMI = st.text_input("BMI value")
    DiabetesPedigreeFunction = st.text_input("Diabetes pedigree function value")
    Age = st.text_input("Age of the person")
    
    
    # code for prediction
    diagnosis = ""
    
    
    # creating a button for predition
    if st.button("Diabetes Test Results"):
        diagnosis = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])

    st.success(diagnosis)
    
    
    
    
if __name__ == "__main__":
    main()