# This example calculates the gradient of a one-weight model manually with NumPy.

import numpy as np


# Store the input, desired output, and model weight as NumPy scalar values.
x = np.array(2.0)
target = np.array(10.0)
w = np.array(3.0)

# The forward pass uses the current weight to make a prediction from the input.
prediction = w * x

# Squared-error loss measures how far the prediction is from the target.
loss = (prediction - target) ** 2

# NumPy performs the calculations above, but it does not automatically track
# their derivatives. For L = (wx - y)^2, the chain rule gives
# dL/dw = 2 * (wx - y) * x. Here, wx is stored in prediction.
# This gradient is dLoss/dw: how the loss changes as the weight changes.
gradient = 2 * (prediction - target) * x

print(f"Input x: {x}")
print(f"Target: {target}")
print(f"Weight: {w}")
print(f"Prediction: {prediction}")
print(f"Loss: {loss}")
print(f"Gradient: {gradient}")
