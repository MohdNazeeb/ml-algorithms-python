# Importing required libraries and dataset
from sklearn.datasets import load_wine
import pandas as pd

# Loading and preparing data
df = load_wine()
dataset = pd.DataFrame(df.data, columns=df.feature_names)
dataset['target'] = df.target

# Separating features and target
x = dataset.drop('target', axis=1)
y = dataset['target']

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)

# Importing Naive Bayes
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()

# Training the model
classifier.fit(X_train, y_train)

# Predictions
y_pred = classifier.predict(X_test)

# Accuracy and classification report
from sklearn.metrics import accuracy_score, classification_report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
