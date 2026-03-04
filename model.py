# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import pandas as pd
import streamlit as st
import joblib

model = joblib.load("rfiris.pkl")


st.title("IRIS FLOWER CLASSIFICATION APPLICATION")


st.write("Predict the species of an Iris Flower Using a Random Forest Using a Random Forest Model")


form = st.form("iris form")

form.subheader("Enter Flower Measurement")

sepal_length = form.number_input(

        "sepal_length(cm)",
         min_value=4.0,
         max_value=8.0,
         value = 5.1
     )


sepal_width = form.number_input(

        "sepal_width(cm)",
         min_value=1.0,
         max_value=4.5,
         value = 5.1
     )


petal_length = form.number_input(

        "petal_length(cm)",
         min_value=1.0,
         max_value=7.0,
         value = 5.1
     )



petal_width = form.number_input(

        "petal_width(cm)",
         min_value=0.1,
         max_value=2.5,
         value = 5.1
     )



submit_button = form.form_submit_button("predict")

if submit_button:
    input_data = pd.DataFrame({
        "sepal_length(cm)":[sepal_length],
        "sepal_width(cm)":[sepal_width],
        "petal_length(cm)":[petal_length],
        "petal_width(cm)":[petal_width]
    })
    prediction = model.predict(input_data)


    st.subheader("prediction Result")
    st.success(f" predicted species: {prediction[0]}")
