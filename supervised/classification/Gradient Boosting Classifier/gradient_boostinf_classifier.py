# Importing required libraries and dataset
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np

# Loading and preparing data
df = load_iris()
dataset = pd.DataFrame(df.data)
dataset.columns = df.feature_names

# Separating features and target
x = dataset
y = df.target

# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

# Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
classifier = GradientBoostingClassifier(random_state=42)

# Hyperparameter tuning using GridSearchCV
from sklearn.model_selection import GridSearchCV
parameters = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [2, 3, 4],
}

gb_cv = GridSearchCV(estimator=classifier, param_grid=parameters, scoring='accuracy', cv=5, verbose=2, n_jobs=-1)
gb_cv.fit(X_train, y_train)

# Best parameters and scores
print("Best Parameters:", gb_cv.best_params_)
print("Best CV Score:", gb_cv.best_score_)
print("Best Estimator:", gb_cv.best_estimator_)

# Prediction and Evaluation
y_pred = gb_cv.predict(X_test)

from sklearn.metrics import accuracy_score, classification_report
print("Test Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
