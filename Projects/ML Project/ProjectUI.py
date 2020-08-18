## Importing Libraries
import numpy as np
import pandas as pd
import streamlit as st
from streamlit import caching

## Title
st.title('Open Source Machine Learning App')

## Adding sidebar and listbox for data view and model selection
uploaded_input_data = st.file_uploader('Upload CSV Data',encoding = 'auto')

if st.button('View Sample Data'):
       if uploaded_input_data is not None:
            input_data = pd.read_csv(uploaded_input_data)
            st.dataframe(input_data.head())
       else:
            st.write('Input file not found')

if st.button('Reset'):
    st.empty()
    caching.clear_cache()                
    del uploaded_input_data

    

