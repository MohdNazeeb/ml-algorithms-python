import pandas as pd
from sklearn.datasets import fetch_california_housing
import numpy as np
from matplotlib import pyplot as plt
df=fetch_california_housing()
dataset=pd.DataFrame(df.data)
dataset.columns=df.feature_names
print(dataset.head())

#Independent features and Dependent features
x=dataset
y=df.target

#train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=42)

#Standardizing the dataset
from sklearn.preprocessing import StandardScaler
scalar=StandardScaler()
X_train=scalar.fit_transform(X_train)
X_test=scalar.transform(X_test)

#Applying Linear Regression
from sklearn.linear_model import LinearRegression
#Cross Validation
from sklearn.model_selection import cross_val_score
regression=LinearRegression()
regression.fit(X_train,y_train)
mse=cross_val_score(regression,X_train,y_train,scoring='neg_mean_squared_error',cv=10)
np.mean(mse)
#Prediction
reg_pred=regression.predict(X_test)

#visualizing the predicting value with actual value
import seaborn as sns
sns.displot(reg_pred-y_test,kind='kde')
plt.show()

