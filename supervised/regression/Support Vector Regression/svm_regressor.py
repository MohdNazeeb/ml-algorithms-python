# Importing required libraries and dataset
from sklearn.datasets import fetch_california_housing
import pandas as pd

# Loading and preparing data
df = fetch_california_housing()
dataset = pd.DataFrame(df.data, columns=df.feature_names)
dataset['target'] = df.target

# Separating features and target
x = dataset.drop('target', axis=1)
y = dataset['target']

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

# Importing SVM Regressor
from sklearn.svm import SVR
regressor = SVR()

# Hyperparameter tuning
from sklearn.model_selection import GridSearchCV
parameters = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'poly', 'rbf'],
    'gamma': ['scale', 'auto']
}
svr_cv = GridSearchCV(regressor, param_grid=parameters, scoring='r2', cv=3)
svr_cv.fit(X_train, y_train)

# Predictions
y_pred = svr_cv.predict(X_test)

# Best parameters and score
print(svr_cv.best_params_)
print(svr_cv.best_score_)

# Regression metrics
from sklearn.metrics import mean_squared_error, r2_score
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))
