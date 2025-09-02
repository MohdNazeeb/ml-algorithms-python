# Importing California housing dataset
from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd

# Loading dataset into DataFrame
df = fetch_california_housing()
dataset = pd.DataFrame(df.data)
dataset.columns = df.feature_names  # Naming the columns

# Separating features and target variable
x = dataset                         # Independent variables
y = df.target                       # Dependent variable

# Splitting into training and testing sets (70% train, 30% test)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

# Standardizing the features (important for regularized models like Ridge)
from sklearn.preprocessing import StandardScaler
scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

# Importing Lasso Regression
from sklearn.linear_model import Lasso

# Importing GridSearchCV for hyperparameter tuning
from sklearn.model_selection import GridSearchCV
lasso_regressor = Lasso()

# Defining range of alpha values (regularization strength) to test
parameter = {'alpha': [1, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80]}

# Performing Grid Search with 5-fold cross-validation and neg MSE as the score
lassocv = GridSearchCV(lasso_regressor, parameter, scoring='neg_mean_squared_error', cv=5)
lassocv.fit(X_train, y_train)

# Printing best alpha value and corresponding score
print(lassocv.best_params_)     # Best regularization value
print(lassocv.best_score_)      # Best cross-validated negative MSE

# Predicting on the test set using the best model
lasso_pred = lassocv.predict(X_test)

# Plotting KDE of residuals to visualize prediction error
import seaborn as sns
import matplotlib.pyplot as plt
sns.displot(lasso_pred - y_test, kind='kde')
plt.title("Distribution of Residuals")
plt.xlabel("Residuals")
plt.show()

# Calculating and printing R² score to check model performance
from sklearn.metrics import r2_score
score = r2_score(y_test, lasso_pred)
print("R² Score:", score)
