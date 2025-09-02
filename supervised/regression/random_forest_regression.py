# Importing California Housing Dataset
from sklearn.datasets import fetch_california_housing
import pandas as pd
import numpy as np

# Load and Prepare dataset
df = fetch_california_housing()
dataset = pd.DataFrame(df.data, columns=df.feature_names)

# Separation of Dependent and Independent variable
X = dataset
y = df.target
# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

# Normalizing or Standardizing (Optional for RandomForest)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor()

# Hyperparameter Tuning using GridSearchCV
from sklearn.model_selection import GridSearchCV
parameters = {
    'max_depth': [2, 3, 4, 5, 6, 7],
    'max_features': ["sqrt", "log2"],
    'random_state': [42, 43, 45, 60, 65],
    'max_samples': [0.5, 0.7, 0.9, None] 
}
random_forest_cv = GridSearchCV(regressor, param_grid=parameters, scoring='neg_mean_squared_error', cv=5)
random_forest_cv.fit(X_train, y_train)

# Best Params and Score
print(random_forest_cv.best_estimator_)
print(random_forest_cv.best_score_)
print(random_forest_cv.best_params_)

# Prediction and Evaluation
y_pred = random_forest_cv.predict(X_test)
from sklearn.metrics import mean_squared_error, r2_score
print(mean_squared_error(y_test, y_pred))
print(r2_score(y_test, y_pred))
