#Impoorting iris dataset
from sklearn.datasets import load_iris
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Load and Prepare dataset
df=load_iris()
dataset=pd.DataFrame(df.data,columns=df.feature_names)
x=dataset
y=df.target

#Train Test split the dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.30,random_state=42)

#Decision Tree Classifier
from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(max_depth=2) #Pruning the Tree
classifier.fit(X_train,y_train)

#Visualizing the tree
from sklearn import tree
plt.figure(figsize=(15,10))
tree.plot_tree(classifier,filled=True)
plt.show()

#Prediction and Evaluation
y_pred=classifier.predict(X_test)
from sklearn.metrics import accuracy_score,classification_report
print(accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
