#Importing Dataset
from sklearn.datasets import fetch_california_housing
import numpy as np
import pandas as pd

#Load and Prepare dataset
df=fetch_california_housing()
dataset=pd.DataFrame(df.data)
dataset.columns=df.feature_names

#Seperation of Dependent and Independent variable
x=dataset
y=df.target

#Train Test Split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=42)

#Decision Tree Regression
from sklearn.tree import DecisionTreeRegressor
regressor=DecisionTreeRegressor(max_depth=5,random_state=42)
regressor.fit(X_train,y_train)
#Visualizing The Tree
from  sklearn.tree import plot_tree
from matplotlib import pyplot as plt
plt.figure(figsize=(15, 10))
plot_tree(
    regressor,
    feature_names=x.columns,
    filled=True,
    rounded=True,
    fontsize=10
)

#Prediction and Evavluation
y_pred=regressor.predict(X_test)
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score
print(mean_squared_error(y_test,y_pred))
print(mean_absolute_error(y_test,y_pred))
print(r2_score(y_test,y_pred))

#Accuracy
import seaborn as sns
sns.displot(y_pred-y_test,kind='kde')
plt.plot()
print("Train R² Score:", r2_score(y_train, regressor.predict(X_train)))
print("Test R² Score:", r2_score(y_test, y_pred))




