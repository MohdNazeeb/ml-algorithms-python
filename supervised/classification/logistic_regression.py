# Importing the required libraries and dataset
from sklearn.datasets import load_wine
import pandas as pd
import numpy as np

# Loading and cleaning the data
df = load_wine()  
dataset = pd.DataFrame(df.data)  
dataset.columns = df.feature_names 
dataset['target'] = df.target  

# Separation of dependent  and independent features
x = dataset.drop('target', axis=1) 
y = dataset['target']  

# Splitting the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

# Importing Logistic Regression model
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()  

# Hyperparameter tuning using GridSearchCV
from sklearn.model_selection import GridSearchCV
parameter = {'penalty': ['l1', 'l2', 'elasticnet'], 'C': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50]}
logistic_cv = GridSearchCV(classifier, param_grid=parameter, scoring='accuracy', cv=5)  # 5-fold cross-validation
logistic_cv.fit(X_train, y_train)  # Fit model with training data

# Predicting values for the test set
y_pred = logistic_cv.predict(X_test)

# Printing best parameters and best cross-validation score
print(logistic_cv.best_params_)
print(logistic_cv.best_score_)

# Calculating and printing accuracy score
from sklearn.metrics import accuracy_score, classification_report
score = accuracy_score(y_test, y_pred)  # Accuracy on test set
print(score)

# Generating and printing classification report
report = classification_report(y_test, y_pred)
print(report)
