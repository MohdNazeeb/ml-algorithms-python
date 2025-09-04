# 📘 Support Vector Regression (SVR)

Support Vector Regression (SVR) is a type of **Support Vector Machine (SVM)** used for regression tasks.  
Instead of predicting discrete classes (like SVM Classifier), SVR predicts **continuous values** while maintaining the core idea of maximizing the margin.  

---

## 🧠 Intuition

The main idea of SVR is to find a function that approximates the data such that the errors are within a certain threshold **ε (epsilon margin)**.  

- Predictions inside the margin are considered “good enough” (no penalty).  
- Predictions outside the margin are penalized using **slack variables**.  

The model focuses on the **support vectors** (critical data points) that define the regression line.

---

## 📊 Hypothesis Function

For SVR, the prediction function is:

$$
f(x) = w^T x + b
$$

Where:
- $w$ = weight vector  
- $b$ = bias  

The function is constrained to stay within an **ε-tube** around the true values.

---

## ⚙️ Cost Function (Objective)

The SVR objective is to minimize both the model complexity and the prediction error outside the ε margin:

$$
\min_{w,b} \ \frac{1}{2} \|w\|^2 + C \sum_{i=1}^{m} (\xi_i + \xi_i^*)
$$

Subject to:

$$
\begin{cases}
y_i - w^T x_i - b \leq \epsilon + \xi_i \\
w^T x_i + b - y_i \leq \epsilon + \xi_i^* \\
\xi_i, \xi_i^* \geq 0
\end{cases}
$$

Where:
- $m$ = number of training examples  
- $\epsilon$ = margin of tolerance (controls sensitivity)  
- $C$ = regularization parameter (trade-off between flatness and error penalty)  
- $\xi_i, \xi_i^*$ = slack variables (penalty for errors beyond ε)

---

## 🔄 Kernel Trick

Like SVM, SVR can use different **kernels** to handle non-linear data:

- **Linear Kernel** → straight regression line  
- **Polynomial Kernel** → curved regression  
- **RBF Kernel (Gaussian)** → highly flexible, fits complex data  

---

## 📈 Visualization

![SVR Example](https://upload.wikimedia.org/wikipedia/commons/7/72/SVM_margin.png)

- The blue line = regression function  
- Dashed lines = ε margin boundaries  
- Support vectors (critical points) determine the line  

---

## 📊 Accuracy

The following graph shows the accuracy (measured via R² score) of the implemented SVR model:

<img src="accuracySVR.png" alt="SVR Accuracy Graph" width="500"/>
