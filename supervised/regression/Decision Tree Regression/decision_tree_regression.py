#Importing Dataset
from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd

#Load and Prepare dataset
df=fetch_california_housing()
dataset=pd.DataFrame(df.data)
dataset.columns=df.feature_names

#Seperation of Dependent and Independent variable
x=dataset
y=df.target

#Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=42)

#Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
regressor=DecisionTreeRegressor()
parameters={
    'criterion' : ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
    'splitter' : ['best','random'],
    'max_depth' : [1,2,3,4,5,6,7,8,9,10],
    'min_samples_split': [2, 5, 10, 20],
    'min_samples_leaf': [1, 2, 5, 10]
}
decision_tree_cv=GridSearchCV(regressor,param_grid=parameters,scoring='neg_mean_squared_error',cv=5)
decision_tree_cv.fit(X_train,y_train)
print(decision_tree_cv.best_params_)
print(decision_tree_cv.best_score_)

#Prediction and Evavluation
y_pred=decision_tree_cv.predict(X_test)
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print(mean_squared_error(y_test,y_pred))
print(mean_absolute_error(y_test,y_pred))
print(r2_score(y_test,y_pred))



