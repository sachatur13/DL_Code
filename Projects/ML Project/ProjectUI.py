## Importing Libraries
import numpy as np
import pandas as pd
import streamlit as st
from streamlit import caching
import LinearRegression as LR

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


st.sidebar.text('Select Model :')
if st.sidebar.checkbox('Linear Regression'):
     st.title('Linear Regression')
     target_variable = st.text_input('Input Target Variable','')    
     test_size = st.slider('Test data size for train test split',0.0,1.0,0.1)
     if target_variable!='':
          train_X,test_X,train_y,test_y = LR.get_training_testing_data(input_data,
                                                                  test_size,
                                                                  target_variable)
          st.write('Training data size: ',train_X.shape)
          st.write('Test data size: ',test_X.shape)
          
          input_data_columns = list(input_data.columns.drop(target_variable))
          selected_column = st.selectbox('Select Variable for scatterplot',(input_data_columns))
          
          plot = LR.test_linear_regression_assumptions(target_variable
                                                       ,selected_column,input_data)
          st.pyplot()
          
     
    

