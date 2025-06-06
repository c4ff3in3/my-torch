import torch
import math

dtype = torch.float
device = torch.device("mps")

# Create Inputs
x = torch.linspace(-math.pi, math.pi, 2000, device=device, dtype=dtype)
y = torch.sin(x)

a = torch.randn((), device=device, dtype=dtype, requires_grad=True)
b = torch.randn((), device=device, dtype=dtype, requires_grad=True)
c = torch.randn((), device=device, dtype=dtype, requires_grad=True)
d = torch.randn((), device=device, dtype=dtype, requires_grad=True)

learning_rate = 1e-6

for t in range(2000):
    y_pred = a + b * x + c * x**2 + d * x**3

    loss = (y_pred - y).pow(2).sum()
    print("Loss", loss)

    # Use autograd to compute the backward pass. This call will compute the
    # gradient of loss with respect to all Tensors with requires_grad=True.
    # After this call a.grad, b.grad. c.grad and d.grad will be Tensors holding
    # the gradient of the loss with respect to a, b, c, d respectively.
    loss.backward()

    # Manually update weights using gradient descent. Wrap in torch.no_grad()
    # because weights have requires_grad=True, but we don't need to track this
    # in autograd.
    with torch.no_grad():
        a -= learning_rate * a.grad
        b -= learning_rate * b.grad
        c -= learning_rate * c.grad
        d -= learning_rate * d.grad

        a.grad = None
        b.grad = None
        c.grad = None
        d.grad = None


"""
 The forward function computes output Tensors from input Tensors. 
 The backward function receives the gradient of the output Tensors with respect to 
 some scalar value, and computes the gradient of the input Tensors with respect to 
"""


print(f"Result: y = {a.item()} + {b.item()}x + {c.item()}x^2 + {d.item()}x^3")
