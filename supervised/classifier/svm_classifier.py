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

# Importing SVM Classifier
from sklearn.svm import SVC
classifier = SVC()

# Hyperparameter tuning
from sklearn.model_selection import GridSearchCV
parameters = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto']
}
svm_cv = GridSearchCV(classifier, param_grid=parameters, scoring='accuracy', cv=5)
svm_cv.fit(X_train, y_train)

# Predictions
y_pred = svm_cv.predict(X_test)

# Best parameters and score
print(svm_cv.best_params_)
print(svm_cv.best_score_)

# Accuracy and classification report
from sklearn.metrics import accuracy_score, classification_report
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
