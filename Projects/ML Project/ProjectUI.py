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
     st.write('Null % in the dataset')
     st.write(input_data.isna().mean()*100)
     if st.checkbox('Impute Nulls'):
          imputed_data = input_data.fillna(method = 'ffill')
          st.write(imputed_data.isna().mean()*100)
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
     test_size = st.slider('Test data size for train test split',0.1,1.0,0.1)
     if target_variable!='':
          train_X,test_X,train_y,test_y = LR.get_training_testing_data(imputed_data,
                                                                  test_size,
                                                                  target_variable)
          st.write('Training data size: ',train_X.shape)
          st.write('Test data size: ',test_X.shape)
          
          st.write('Test for Linear Model Validation')
          
          if st.checkbox('Linearity Test'):
               test_type = 'Linearity'
               input_data_columns = list(imputed_data.columns.drop(target_variable))
          #st.write(input_data_columns)
               selected_column = st.selectbox('Select columns for plot',(input_data_columns))
               st.write(selected_column)
               plot = LR.test_linear_regression_assumptions(target_variable
                                                       ,selected_column,input_data
                                                       ,test_type,
                                                       None)
               st.pyplot()
          
          if st.checkbox('Multicollinearity Check'):
               test_type = 'Multicollinearity'
               corr_plot = LR.test_linear_regression_assumptions(None,None,imputed_data,
                                                                test_type,
                                                                 None,None) 
               st.pyplot()
               
          if st.checkbox('Mean Residual Checks'):
               test_type = 'Residual Mean'
               regressor,prediction_y,r2 = LR.fit_linear_regression_model(train_X,test_X,
                                                                       train_y,test_y)
               st.write('R2 for the fitted model : ',r2)
                              
               residual = LR.test_linear_regression_assumptions(None,None,None,
                                                                test_type,
                                                                 test_y,prediction_y) 
               
               st.write('Residual mean for the fitted model : ',residual)

          if st.checkbox('Test for Homoscedasticity'):
               test_type = 'Homoscedasticity'
               
               regressor,prediction_y,r2 = LR.fit_linear_regression_model(train_X,test_X,
                                                                       train_y,test_y)
               residualvsfitted_plot = LR.test_linear_regression_assumptions(None,None,None,
                                                                test_type,
                                                                 test_y,prediction_y)
               
               st.pyplot()
               
               
          if st.checkbox('Normality of Error Terms'):
               test_type = 'Normality'
               
               regressor,prediction_y,r2 = LR.fit_linear_regression_model(train_X,test_X,
                                                                       train_y,test_y)
               error_plot = LR.test_linear_regression_assumptions(None,None,None,
                                                                test_type,
                                                                 test_y,prediction_y)
               
               st.pyplot()

