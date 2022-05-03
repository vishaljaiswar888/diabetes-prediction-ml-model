# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle   



# loading the saved model
loaded_model = pickle.load(open("D:/ProgramData/ML_Siddhardhan/1. Project Diabetes/diabetes_trained_model.sav", "rb"))



input_data = (8,99,84,0,0,35.4,0.388,50)

# converting the input_data into numpy array
input_data_as_np_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
  # why reshaping : because our model is trained on 768 instances with 8 columns, so our model expects same
  # amount of data to perform the same task 
input_data_reshaped = input_data_as_np_array.reshape(1, -1)


# we can skip the [ standardize the input data step ], because here currently we are having the small
# small dataset, but we have to use it while working on bigdatas.


prediction = loaded_model.predict(input_data_reshaped)
print(prediction)

if (prediction[0]==0):
  print("The person is not diabetic.")
else:
  print("The person is diabetic.")