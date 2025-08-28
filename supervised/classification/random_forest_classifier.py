#Importing required liabraries and datasets
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

# Loading and preparing data
df=load_iris()
dataset=pd.DataFrame(df.data )
dataset.columns=df.feature_names

# Separating features and targete
x=dataset
y=df.target

#Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=42)

#Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
classiifier=RandomForestClassifier()

#Hyper parameter tuning using - GridSearchCV
from sklearn.model_selection import GridSearchCV
parameters = {
    'n_estimators': [100, 300, 500, 1000],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2']
}
random_forest_cv=GridSearchCV(classiifier,param_grid=parameters,scoring='accuracy',cv=5)
random_forest_cv.fit(X_train,y_train)

#Best parameters and scores
print(random_forest_cv.best_params_)
print(random_forest_cv.best_score_)
print(random_forest_cv.best_estimator_)

#Prediction and Evaluation
y_pred=random_forest_cv.predict(X_test)
from sklearn.metrics import accuracy_score,classification_report
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
