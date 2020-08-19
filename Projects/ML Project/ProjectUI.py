## Importing Libraries
import numpy as np
import pandas as pd
import streamlit as st
from streamlit import caching

## Title
st.title('Open Source Machine Learning App')

## Adding sidebar and listbox for data view and model selection
uploaded_input_data = st.file_uploader('Upload CSV Data',encoding = 'auto')
if uploaded_input_data is not None:
     input_data = pd.read_csv(uploaded_input_data)
else:
     st.write('Input file not found')
     
def get_column_definition(input_data):
     numerical_input_columns = input_data.select_dtypes(include = ['int','float32']).columns
     categorical_input_columns = input_data.select_dtypes(include = ['O']).columns
     date_input_columns = input_data.select_dtypes(include = ['datetime']).columns
     
     return numerical_input_columns,categorical_input_columns,date_input_columns

st.sidebar.title('Use these options for analysis')
if st.sidebar.checkbox('View Sample Data'):
     st.dataframe(input_data.head())
     n,c,d = get_column_definition(input_data)
     
if st.sidebar.checkbox('View Dataset details'):
     st.write(input_data.info())

if st.sidebar.checkbox('Plot data'):
     st.bar_chart(input_data)

st.sidebar.text('Select Model :')
if st.sidebar.checkbox('Linear Regression'):
     st.write('You Selected Linear Regression')

    

