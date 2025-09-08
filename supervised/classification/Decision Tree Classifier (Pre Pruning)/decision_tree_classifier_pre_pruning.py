#Importing iris dataset
from sklearn.datasets import load_iris
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Load and Prepare the dataset
df=load_iris()
dataset=pd.DataFrame(df.data,columns=df.feature_names)
dataset['target']=df.target

#Seperation of Dependent ands Independent variable
x=dataset.drop('target',axis=1)
y=dataset['target']

#Train Test Splitting the dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=42)

#Decision Tree
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier()

#hyperparameter tuning using GridSearchCV
from sklearn.model_selection import GridSearchCV
parameters={
    'criterion' : ['gini','entropy','log_loss'],
    'splitter': ['best','random'],
    'max_depth' : [1,2,3,4,5,6,7,8],
    'max_features' : [1,2,3,4,5,6,7]
}
tree_cv=GridSearchCV(classifier,param_grid=parameters,scoring='accuracy',cv=5)
tree_cv.fit(X_train,y_train)

#Best Parameters and Scores
print(tree_cv.best_estimator_)
print(tree_cv.best_params_)
print(tree_cv.best_score_)

#Prediction and Evaluation
y_pred=tree_cv.predict(X_test)
from sklearn.metrics import accuracy_score,classification_report
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))


