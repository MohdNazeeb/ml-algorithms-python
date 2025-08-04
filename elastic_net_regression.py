# Importing California Housing Dataset
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np

# Load and Prepare Dataset
df = fetch_california_housing()
dataset = pd.DataFrame(df.data, columns=df.feature_names)
x = dataset
y = df.target

# Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ElasticNet Regression with GridSearchCV
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import GridSearchCV

parameters = {
    'alpha': [1, 2, 3, 5, 10, 20, 30, 40],
    'l1_ratio': [0.1, 0.2, 0.3, 0.5, 0.7, 1]
}

elastic_regressor = ElasticNet(max_iter=10000)
elastic_cv = GridSearchCV(elastic_regressor, param_grid=parameters, scoring='neg_mean_squared_error', cv=5)
elastic_cv.fit(X_train, y_train)

# Best Params and Score
print("Best Parameters:", elastic_cv.best_params_)
print("Best Score (Neg MSE):", elastic_cv.best_score_)

# Prediction and Evaluation
from sklearn.metrics import mean_squared_error, r2_score
elastic_pred = elastic_cv.predict(X_test)
print("Test MSE:", mean_squared_error(y_test, elastic_pred))
print("R² Score:", r2_score(y_test, elastic_pred))
