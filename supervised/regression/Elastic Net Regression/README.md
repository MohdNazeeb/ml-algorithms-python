# 📘 Elastic Net Regression

Elastic Net Regression is a **regularized regression technique** that combines both **L1 (Lasso)** and **L2 (Ridge)** penalties.  
It is useful when we want the benefits of **feature selection (Lasso)** and **coefficient shrinkage (Ridge)** at the same time.  

---

##  Intuition

The main idea is to balance between **Lasso** and **Ridge** by controlling their contributions using a parameter.  
- Lasso encourages sparsity (some coefficients become 0).  
- Ridge shrinks coefficients but keeps all features.  
- Elastic Net blends both approaches.

For **Elastic Net Regression**:

$$
h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + ... + \theta_n x_n
$$

---

##  Cost Function

The Elastic Net cost function is:

$$
J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda_1 \sum_{j=1}^{n} |\theta_j| + \lambda_2 \sum_{j=1}^{n} \theta_j^2
$$

Where:
- $m$ = number of training examples  
- $h_\theta(x^{(i)})$ = predicted value  
- $y^{(i)}$ = actual value  
- $\lambda_1$ = L1 regularization parameter (Lasso part)  
- $\lambda_2$ = L2 regularization parameter (Ridge part)  
- $\theta_j$ = model parameters (excluding $\theta_0$)   

⚠️ Note:  
- Larger $\lambda_1$ → stronger sparsity (more zeros in coefficients).  
- Larger $\lambda_2$ → stronger shrinkage (smaller coefficients).  

---

##  Gradient Descent

The update rule becomes:

$$
\theta_j := \theta_j - \alpha \left( \frac{1}{m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)} + \lambda_1 \cdot sign(\theta_j) + \lambda_2 \theta_j \right)
$$

Where:  
- $\alpha$ = learning rate  
- $\lambda_1 \cdot sign(\theta_j)$ = Lasso shrinkage term (L1)  
- $\lambda_2 \theta_j$ = Ridge shrinkage term (L2)  

---

##  Visualization

![Elastic Net Regression](https://upload.wikimedia.org/wikipedia/commons/thumb/b/b5/Regularization_penalties.png/640px-Regularization_penalties.png)

- Elastic Net combines the strengths of both Ridge and Lasso.  
- It is especially useful when there are **many correlated features**.  

---

##  Accuracy

The following graph shows the accuracy of the implemented Elastic Net Regression model:

<img src="accuracyElasticNet.png" alt="Accuracy Graph" width="500"/>
