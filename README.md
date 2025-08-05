# 🧠 Machine Learning Algorithms in Python

Welcome to my **Machine Learning** repository! This repo contains a collection of classic ML algorithms implemented using **Python**, primarily using libraries like **scikit-learn**, **NumPy**, **Pandas**, and **Matplotlib**. Each model includes basic theory, implementation, and visualizations where applicable.

## Algorithms Implemented

| Type              | Algorithms                         |
|-------------------|------------------------------------|
| 🔵 Supervised     | Linear Regression, Logistic Regression, Decision Tree, K-Nearest Neighbors (KNN), Support Vector Machine (SVM) |
| 🟠 Unsupervised   | K-Means Clustering                 |
| 🟣 Evaluation     | Accuracy, Precision, Recall, F1-Score, Confusion Matrix |
| 🟢 Coming Soon    | Random Forest, PCA, Naive Bayes    |


##  Datasets Used

All models are tested using standard datasets from `sklearn.datasets` and `CSV` files:

- **Iris Dataset**
- **Diabetes Dataset**
- **Boston Housing Dataset**
- **Custom CSVs in `datasets/` folder**

You can easily load them using:

```python
from sklearn.datasets import load_iris
data = load_iris()


