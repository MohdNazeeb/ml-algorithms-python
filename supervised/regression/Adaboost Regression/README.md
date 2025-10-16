# 📘 AdaBoost Regression

**AdaBoost Regression** (Adaptive Boosting for Regression) is an **ensemble learning technique** that combines multiple **weak learners** (usually decision trees) to create a strong **regression model**.  
It focuses more on the **instances with higher errors** during training, adapting iteratively to improve predictions.

---

## Intuition

The main idea is to **sequentially train models**, where each new model **tries to correct the errors** made by the previous ones.  
Each learner is trained on a modified version of the data, giving more weight to instances with **higher prediction errors**.

In the end, predictions are **combined** (often through a weighted average) to produce the final output.

---

## Model Structure

AdaBoost for regression works by:

1. Fitting a weak learner to the data.
2. Computing the prediction error.
3. Increasing the weights of poorly predicted samples.
4. Repeating steps 1–3 for a set number of iterations or until performance stops improving.
5. Aggregating the predictions from all weak learners using a **weighted sum**.

---

## Loss Function

AdaBoost Regression often uses the **exponential loss function** or **squared loss**, depending on the implementation. A general form is:

$$
L(y, \hat{y}) = \sum_{i=1}^{m} w_i \cdot (y^{(i)} - \hat{y}^{(i)})^2
$$

Where:
- \( y^{(i)} \) = actual target value  
- \( \hat{y}^{(i)} \) = predicted value from the ensemble  
- \( w_i \) = weight of the \(i\)-th training instance (increases with higher error)  
- \( m \) = number of training examples  

⚠️ Note: Samples with higher error in previous rounds get **higher weight**, pushing the next learner to focus more on them.

---

## Training Process

At each iteration \( t \):

1. Fit a weak learner \( h_t(x) \) to the data.
2. Compute error:
   $$
   \epsilon_t = \frac{\sum_{i=1}^{m} w_i^{(t)} \cdot |y^{(i)} - h_t(x^{(i)})|}{\sum_{i=1}^{m} w_i^{(t)}}
   $$
3. Compute model weight:
   $$
   \alpha_t = \ln\left(\frac{1 - \epsilon_t}{\epsilon_t}\right)
   $$
4. Update sample weights:
   $$
   w_i^{(t+1)} = w_i^{(t)} \cdot \exp(\alpha_t \cdot |y^{(i)} - h_t(x^{(i)})|)
   $$
5. Normalize weights.

The final prediction is a **weighted sum** of all weak learners:

$$
\hat{y}(x) = \sum_{t=1}^{T} \alpha_t \cdot h_t(x)
$$

---

## Visualization

<img src="adaboost_diagram.jpeg" alt="Adaboost Visualization" width="500"/>

- Each learner focuses on the **mistakes of its predecessors**.  
- Final prediction is a **combination** of all learners’ outputs.  
- Helps reduce **bias** and **variance**.

---

## Accuracy

The following graph shows the accuracy (or performance metrics) of the implemented AdaBoost Regression model:

<img src="accuracyAdaBoost.png" alt="Accuracy Graph" width="500"/>
