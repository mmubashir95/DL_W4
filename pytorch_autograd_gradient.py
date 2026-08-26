# This example uses Autograd and applies one manual SGD weight update in PyTorch.

import torch


# Only w needs requires_grad=True because it is the value whose gradient we
# want. The input and target are fixed values, so their gradients are not needed.
x = torch.tensor(2.0)
target = torch.tensor(10.0)
w = torch.tensor(
    3.0,
    requires_grad=True,
)
learning_rate = 0.01

# During this forward pass, PyTorch builds a computational graph that records
# how the prediction and loss depend on w.
prediction = w * x
loss = (prediction - target) ** 2

# backward() performs backpropagation and calculates dLoss/dw. PyTorch
# automatically applies the same derivative written manually in the NumPy
# example, and stores the result in w.grad.
loss.backward()

# Save these values before changing w so the before-and-after values can both be
# displayed. The gradient tells us how the loss changes with respect to w.
original_weight = w.item()
gradient = w.grad.item()

# PyTorch normally tracks tensor operations for Autograd. The update should not
# become part of that computational graph, so perform it inside no_grad(). This
# is conceptually the update that an SGD optimizer will later perform for us.
with torch.no_grad():
    w -= learning_rate * w.grad

# Gradient is negative, so subtracting it increases w. Check the model again to
# verify that this update moved the weight in a direction that reduces the loss.
with torch.no_grad():
    new_prediction = w * x
    new_loss = (new_prediction - target) ** 2

print(f"Input x: {x.item()}")
print(f"Target: {target.item()}")
print(f"Original Weight: {original_weight}")
print(f"Prediction Before Update: {prediction.item()}")
print(f"Loss Before Update: {loss.item()}")
print(f"Gradient: {gradient}")
print(f"Learning Rate: {learning_rate}")
print(f"Updated Weight: {w.item():.2f}")
print(f"Prediction After Update: {new_prediction.item():.2f}")
print(f"Loss After Update: {new_loss.item():.4f}")
