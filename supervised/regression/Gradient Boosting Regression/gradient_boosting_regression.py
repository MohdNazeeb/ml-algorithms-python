# Importing required libraries and dataset
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np

# Loading and preparing data
df = fetch_california_housing()
dataset = pd.DataFrame(df.data)
dataset.columns = df.feature_names

# Separating features and target
x = dataset
y = df.target

# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

# Gradient Boosting Regressor
from sklearn.ensemble import GradientBoostingRegressor
regressor = GradientBoostingRegressor(random_state=42)

# Hyperparameter tuning using GridSearchCV
from sklearn.model_selection import GridSearchCV
parameters = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [2, 3, 4]
}

gb_reg_cv = GridSearchCV(estimator=regressor, param_grid=parameters, scoring='r2', cv=5, verbose=2, n_jobs=-1)
gb_reg_cv.fit(X_train, y_train)

# Best parameters and scores
print("Best Parameters:", gb_reg_cv.best_params_)
print("Best CV R2 Score:", gb_reg_cv.best_score_)
print("Best Estimator:", gb_reg_cv.best_estimator_)

# Prediction and Evaluation
y_pred = gb_reg_cv.predict(X_test)

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
print("R2 Score:", r2_score(y_test, y_pred))
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_pred))
