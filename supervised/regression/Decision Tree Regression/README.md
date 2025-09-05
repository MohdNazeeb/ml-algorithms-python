# 🌳 Decision Tree Regression

Decision Tree Regression is a type of **supervised learning algorithm** used for regression tasks.  
Unlike linear models, Decision Trees split the data into **regions based on feature thresholds** and predict the **mean value** of the target in each region.  
This makes them powerful for capturing **non-linear relationships** between input features and continuous outputs.

---

## 🧠 Intuition

The core idea is to **partition the feature space** into smaller regions such that the variance of the target values within each region is minimized.  

- At each node, the algorithm chooses the **best feature and threshold** that minimizes prediction error.  
- The dataset is split recursively, forming a tree structure.  
- Final predictions are made at the **leaf nodes** by taking the mean of the target values in that region.  

---

## 📊 Hypothesis Function

For Decision Tree Regression, the prediction for a sample $x$ is:

$$
h(x) = \frac{1}{N_t} \sum_{i \in R_t} y_i
$$

Where:
- $R_t$ = region of the leaf node containing $x$  
- $N_t$ = number of samples in that region  
- $y_i$ = target values of samples in that region  

---

## ⚙️ Cost Function (Objective)

The objective is to split the data such that the **Mean Squared Error (MSE)** is minimized.  

For a node $t$, the impurity is:

$$
MSE(t) = \frac{1}{N_t} \sum_{i \in R_t} (y_i - \bar{y_t})^2
$$

Where:
- $N_t$ = number of samples in node $t$  
- $\bar{y_t}$ = mean of target values in node $t$  

The split is chosen to minimize the weighted average of the child node impurities.

---

## 🔄 Splitting Process

1. For each feature, evaluate all possible thresholds.  
2. Compute the **reduction in MSE** for each split.  
3. Select the split with the **maximum error reduction**.  
4. Repeat recursively until stopping criteria are met (e.g., max depth, min samples per leaf).  

---

## 📈 Visualization and Terminologies

<img src="decision_tree_regression_image.png" alt="Decision Tree Terminologies" width="500"/>

- Each **rectangle** (node) corresponds to a region of feature space.  
- Splits are made based on threshold conditions (e.g., $x_j \leq t$).  
- Predictions in the leaf nodes are the **mean target values** of that region.  
- The tree recursively partitions the data until constraints are met.  

---

## ✅ Advantages

- Captures **non-linear** relationships  
- Easy to **interpret and visualize**  
- Requires little to no **feature scaling or preprocessing**

---

## ❌ Limitations

- High risk of **overfitting** if not pruned  
- Small changes in data can drastically change the tree (high variance)  
- Often outperformed by **ensemble methods** (Random Forest, Gradient Boosted Trees)

---

