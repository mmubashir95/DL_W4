import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split


# A fixed seed makes the generated data and initial model weights reproducible.
torch.manual_seed(42)


# -----------------------------------------------------------------------------
# 1. Create a small synthetic regression dataset
# -----------------------------------------------------------------------------
number_of_samples = 256
number_of_features = 10

X = torch.randn(number_of_samples, number_of_features)  # Shape: [256, 10]

# Create targets from a learnable linear relationship, plus a little noise.
true_weights = torch.tensor(
    [[2.0], [-1.0], [0.5], [3.0], [-2.5], [1.5], [0.0], [0.8], [-0.3], [1.2]]
)
true_bias = 0.7
y = X @ true_weights + true_bias + 0.1 * torch.randn(number_of_samples, 1)
# y shape: [256, 1]


# -----------------------------------------------------------------------------
# 2. Split the dataset and create DataLoaders
# -----------------------------------------------------------------------------
# Each item in the complete dataset is one (features, target) pair.
complete_dataset = TensorDataset(X, y)

# Use 80% of the examples for training and the remaining 20% for validation.
number_of_training_examples = int(0.8 * len(complete_dataset))
number_of_validation_examples = (
    len(complete_dataset) - number_of_training_examples
)

# A separately seeded generator makes this particular split reproducible.
split_generator = torch.Generator().manual_seed(42)
train_dataset, validation_dataset = random_split(
    complete_dataset,
    [number_of_training_examples, number_of_validation_examples],
    generator=split_generator,
)

batch_size = 32

# Shuffle training batches so the model sees examples in a different order.
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
)

# Validation only measures performance, so its example order does not matter.
validation_loader = DataLoader(
    validation_dataset,
    batch_size=batch_size,
    shuffle=False,
)


# -----------------------------------------------------------------------------
# 3. Define the multilayer perceptron
# -----------------------------------------------------------------------------
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(10, 64)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.output_layer = nn.Linear(32, 1)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.output_layer(x)
        return x


model = MLP()
print("Model architecture:")
print(model)


# MSELoss measures the difference between the prediction and the true target.
loss_fn = nn.MSELoss()

# model.parameters() provides the trainable weights and biases to the optimizer.
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
)


# -----------------------------------------------------------------------------
# 4. Train and validate the model
# -----------------------------------------------------------------------------
number_of_epochs = 20

# Set this to True to print shapes and one gradient shape for the first batch.
debug_first_batch = True

training_loss_history = []
validation_loss_history = []

for epoch in range(number_of_epochs):
    # -------------------------------------------------------------------------
    # Training phase starts here
    # -------------------------------------------------------------------------
    # Enable training behavior for layers that act differently during training.
    model.train()

    running_training_loss = 0.0
    number_of_training_examples = 0

    for batch_index, (x_batch, y_batch) in enumerate(train_loader):
        # Forward pass: produce predictions from the current model parameters.
        predictions = model(x_batch)

        # Compare predictions with the correct targets.
        loss = loss_fn(predictions, y_batch)

        # PyTorch accumulates gradients by default, so clear old gradients before
        # computing gradients for the current batch.
        optimizer.zero_grad()

        # Compute gradients of the loss with respect to all trainable parameters.
        loss.backward()

        # Optional debugging: inspect one batch and one parameter's gradient.
        # This runs only once, so it does not clutter every batch's output.
        if debug_first_batch and epoch == 0 and batch_index == 0:
            print("\nFirst training batch shapes:")
            print("x_batch shape:    ", x_batch.shape)       # [32, 10]
            print("y_batch shape:    ", y_batch.shape)       # [32, 1]
            print("predictions shape:", predictions.shape)   # [32, 1]
            print(
                "layer1 weight gradient shape:",
                model.layer1.weight.grad.shape,
            )

        # Use the calculated gradients to update the model's weights and biases.
        optimizer.step()

        # MSELoss returns the mean loss for this batch. Multiplying by the batch
        # size converts it to a total, so a smaller final batch is weighted fairly.
        current_batch_size = x_batch.size(0)
        running_training_loss += loss.item() * current_batch_size
        number_of_training_examples += current_batch_size

    average_training_loss = (
        running_training_loss / number_of_training_examples
    )
    # -------------------------------------------------------------------------
    # Training phase ends here
    # -------------------------------------------------------------------------

    # -------------------------------------------------------------------------
    # Validation phase starts here
    # -------------------------------------------------------------------------
    # Switch the model to evaluation behavior before measuring validation loss.
    model.eval()

    running_validation_loss = 0.0
    number_of_validation_examples = 0

    # Validation does not update the model, so gradient tracking is unnecessary.
    with torch.no_grad():
        for x_batch, y_batch in validation_loader:
            # Forward pass only: produce predictions for the validation batch.
            predictions = model(x_batch)

            # Measure validation error without changing the model parameters.
            validation_loss = loss_fn(predictions, y_batch)

            current_batch_size = x_batch.size(0)
            running_validation_loss += (
                validation_loss.item() * current_batch_size
            )
            number_of_validation_examples += current_batch_size

    average_validation_loss = (
        running_validation_loss / number_of_validation_examples
    )
    # -------------------------------------------------------------------------
    # Validation phase ends here
    # -------------------------------------------------------------------------

    # Store both losses so they can be inspected or plotted in a later lesson.
    training_loss_history.append(average_training_loss)
    validation_loss_history.append(average_validation_loss)

    print(
        f"Epoch {epoch + 1}/{number_of_epochs} | "
        f"Training Loss: {average_training_loss:.4f} | "
        f"Validation Loss: {average_validation_loss:.4f}"
    )


# -----------------------------------------------------------------------------
# 5. Verify a few predictions after training
# -----------------------------------------------------------------------------
model.eval()

# Disable gradient tracking because verification does not update parameters.
with torch.no_grad():
    sample_features = X[:5]
    sample_targets = y[:5]
    sample_predictions = model(sample_features)

print("\nPredictions compared with targets:")
for prediction, target in zip(sample_predictions, sample_targets):
    print(
        f"Prediction: {prediction.item():8.4f} | "
        f"Target: {target.item():8.4f}"
    )
