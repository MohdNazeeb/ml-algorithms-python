#Importing California Housing Feature
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np

#Cleaning the data
df=fetch_california_housing()
dataset=pd.DataFrame(df.data)
dataset.columns=df.feature_names

#Seperation of dependent and independent variables
x=dataset
y=df.target

#Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=42)

#Normalazing the dataset
from sklearn.preprocessing import StandardScaler
scalar=StandardScaler()
scalar.fit_transform(X_train)
scalar.transform(X_test)
