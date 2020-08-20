## Importing Linear Regression Library from sklearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import seaborn as sns


def get_training_testing_data(input_data,test_size,y):
    X = input_data.drop(y,axis = 1)
    y = input_data[y]
    train_X,test_X,train_y,test_y = train_test_split(X,y
                                                     ,test_size=test_size
                                                     ,random_state = 42)
    return train_X,test_X,train_y,test_y

def test_linear_regression_assumptions(y,selected_column,input_data):
    
    ## linear_assumption_1. Linear Relationship between variables.
    ## Tested using scatterplot
    
    scatter_plot = sns.scatterplot(selected_column,y,data = input_data)
    
    return scatter_plot
