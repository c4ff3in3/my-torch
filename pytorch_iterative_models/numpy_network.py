import numpy as np
import math


# Input Data
x = np.linspace(-math.pi, math.pi, 2000)
y = np.sin(x)


# Random Weights
a = np.random.randn()
b = np.random.randn()
c = np.random.randn()
d = np.random.randn()


learning_rate = 1e-6

for t in range(2000):
    # Generate Prediction
    y_pred = a + b * x + c * x**2 + d * x**3

    # Compute the loss
    loss = np.square(y_pred - y).sum()
    print("Loss", loss)

    # Backpropagation
    grad_y_pred = 2.0 * (y_pred - y)
    grad_a = grad_y_pred.sum()
    grad_b = (grad_y_pred * x).sum()
    grad_c = (grad_y_pred * x**2).sum()
    grad_d = (grad_y_pred * x**3).sum()

    # Weight Adjustment
    a -= learning_rate * grad_a
    b -= learning_rate * grad_b
    c -= learning_rate * grad_c
    d -= learning_rate * grad_d

print(f"Result: y = {a} + {b}x + {c}x^2 + {d}x^3")
