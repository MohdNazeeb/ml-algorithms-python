<details>
<summary>📘 Gradient Boosting Regression</summary>

**Gradient Boosting Regression** is an **ensemble learning technique** that builds a strong predictive model by **sequentially combining multiple weak learners** (typically decision trees).  
Unlike AdaBoost, Gradient Boosting focuses on **minimizing the residual errors** of the previous model by using **gradient descent optimization**.

---

## Intuition

The core idea is to **build models one after another**, where each new model tries to **reduce the errors (residuals)** made by the previous models.  
Instead of assigning weights to samples like AdaBoost, Gradient Boosting fits each new learner to the **negative gradient of the loss function** — hence the name *Gradient Boosting*.

Finally, all learners’ predictions are **summed up** to give the final regression output.

---

## Model Structure

Gradient Boosting works as follows:

1. Start with an initial prediction (often the mean of target values).  
2. Compute residuals — the difference between actual and predicted values.  
3. Train a weak learner on these residuals.  
4. Add the learner’s predictions (scaled by a learning rate) to the ensemble.  
5. Repeat steps 2–4 for multiple iterations until the loss stops improving.

The final model is a **weighted sum** of all weak learners.

---

## Loss Function

Gradient Boosting can use various differentiable loss functions such as **Mean Squared Error (MSE)** for regression tasks:

$$
L(y, \hat{y}) = \frac{1}{2} \sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2
$$

Where:  
- \( y^{(i)} \) = actual target value  
- \( \hat{y}^{(i)} \) = predicted value from the ensemble  
- \( m \) = total number of samples  

⚙️ The algorithm minimizes this loss using **gradient descent** to find the direction of maximum error reduction.

---

## Training Process

At each iteration \( t \):

1. Compute the residuals (negative gradient of the loss):
   $$
   r_i^{(t)} = -\frac{\partial L(y_i, \hat{y}_i^{(t-1)})}{\partial \hat{y}_i^{(t-1)}}
   $$
2. Fit a weak learner \( h_t(x) \) to the residuals \( r_i^{(t)} \).
3. Compute the optimal multiplier (step size) \( \gamma_t \) that minimizes the loss:
   $$
   \gamma_t = \arg\min_\gamma \sum_{i=1}^{m} L(y_i, \hat{y}_i^{(t-1)} + \gamma \cdot h_t(x_i))
   $$
4. Update the model predictions:
   $$
   \hat{y}_i^{(t)} = \hat{y}_i^{(t-1)} + \eta \cdot \gamma_t \cdot h_t(x_i)
   $$
   where \( \eta \) is the **learning rate** (controls how much each new learner contributes).
5. Repeat for T iterations.

The final model is:

$$
F_T(x) = F_0(x) + \sum_{t=1}^{T} \eta \cdot \gamma_t \cdot h_t(x)
$$

---

## Visualization

<img src="gradientBoosting.png" alt="Gradient Boosting Visualization" width="500"/>

- Each new model learns from the **residual errors** of previous models.  
- Models are **added sequentially**, each improving upon the last.  
- The process follows **gradient descent** to minimize overall loss.  

---

## Accuracy

The following graph shows the performance of the implemented Gradient Boosting Regression model:

<img src="accuracyGradientBoost.png" alt="Accuracy Graph" width="500"/>

</details>
