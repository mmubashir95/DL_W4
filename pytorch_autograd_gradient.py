# This example uses PyTorch Autograd to find the gradient of a one-weight model.

import torch


# Only w needs requires_grad=True because it is the value whose gradient we
# want. The input and target are fixed values, so their gradients are not needed.
x = torch.tensor(2.0)
target = torch.tensor(10.0)
w = torch.tensor(3.0, requires_grad=True)

# During this forward pass, PyTorch builds a computational graph that records
# how the prediction and loss depend on w.
prediction = w * x
loss = (prediction - target) ** 2

# backward() performs backpropagation through the computational graph.
# PyTorch automatically applies the same derivative written manually in the
# NumPy example. The resulting w.grad value contains dLoss/dw.
loss.backward()

print(f"Input x: {x.item()}")
print(f"Target: {target.item()}")
print(f"Weight: {w.item()}")
print(f"Prediction: {prediction.item()}")
print(f"Loss: {loss.item()}")
print(f"Gradient: {w.grad.item()}")
