# This example calculates a gradient and applies one manual SGD update in NumPy.

import numpy as np


# Store the input, desired output, and model weight as NumPy scalar values.
x = np.array(2.0)
target = np.array(10.0)
w = np.array(3.0)
learning_rate = 0.01

# The forward pass uses the current weight to make a prediction from the input.
prediction = w * x

# Squared-error loss measures how far the prediction is from the target.
loss = (prediction - target) ** 2

# NumPy performs the calculations above, but it does not automatically track
# their derivatives. For L = (wx - y)^2, the chain rule gives
# dL/dw = 2 * (wx - y) * x. Here, wx is stored in prediction.
# This gradient is dLoss/dw: how the loss changes as the weight changes.
gradient = 2 * (prediction - target) * x

# SGD updates a parameter with: weight = weight - learning_rate * gradient.
# The learning rate controls the size of the update. Because this gradient is
# negative, subtracting it increases the weight, which should reduce the loss.
# Store the result separately so the original weight remains available.
updated_w = w - learning_rate * gradient

# Use the updated weight for another forward pass to verify that the loss fell.
new_prediction = updated_w * x
new_loss = (new_prediction - target) ** 2

print(f"Input x: {x}")
print(f"Target: {target}")
print(f"Original Weight: {w}")
print(f"Prediction Before Update: {prediction}")
print(f"Loss Before Update: {loss}")
print(f"Gradient: {gradient}")
print(f"Learning Rate: {learning_rate}")
print(f"Updated Weight: {updated_w:.2f}")
print(f"Prediction After Update: {new_prediction:.2f}")
print(f"Loss After Update: {new_loss:.4f}")
