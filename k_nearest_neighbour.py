from sklearn.datasets import load_wine
import pandas as pd
import numpy as np

# Loading and cleaning the data
df = load_wine()
dataset = pd.DataFrame(df.data)
dataset.columns = df.feature_names
dataset['target'] = df.target

# Separation of dependent and independent features
X = dataset.drop('target', axis=1)
y = dataset['target']

# Splitting the dataset into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Importing KNN model
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier()

# Hyperparameter tuning using GridSearchCV
from sklearn.model_selection import GridSearchCV
parameter = {
    'n_neighbors': [3, 5, 7, 9, 11], 
    'weights': ['uniform', 'distance'],  
    'metric': ['euclidean', 'manhattan', 'minkowski']  
}
# 5-fold CV
knn_cv = GridSearchCV(classifier, param_grid=parameter, scoring='accuracy', cv=5)  
# Fit model with training data
knn_cv.fit(X_train, y_train)  

# Predicting values for the test set
y_pred = knn_cv.predict(X_test)

# Printing best parameters and best cross-validation score
print("Best Parameters:", knn_cv.best_params_)
print("Best Cross-Validation Score:", knn_cv.best_score_)

# Calculating and printing accuracy score
from sklearn.metrics import accuracy_score, classification_report
score = accuracy_score(y_test, y_pred)
print("Accuracy on Test Set:", score)

# Generating and printing classification report
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)
