import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Define the neural network architecture
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Define the training loop
def train_loop(model, optimizer, loss_fn, x_train, y_train, epochs):
    for epoch in range(epochs):
        # Forward pass
        y_pred = model(x_train)
        loss = loss_fn(y_pred, y_train)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress
        if epoch % 100 == 0:
            print('Epoch:', epoch, 'Loss:', loss.item())

# Generate some training data
x_train = torch.tensor([[0, 1], [1, 1], [1, 0], [0, 0]], dtype=torch.float32)
y_train = torch.tensor([[1], [0], [1], [0]], dtype=torch.float32)

# Define the neural network
input_size = 2
hidden_size = 4
output_size = 1
model = NeuralNetwork(input_size, hidden_size, output_size)

# Define the loss function and optimizer
learning_rate = 0.1
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Train the neural network
epochs = 1000
train_loop(model, optimizer, loss_fn, x_train, y_train, epochs)

# Make a prediction on a new input
x_test = torch.tensor([[1, 0]], dtype=torch.float32)
y_pred = model(x_test)
print('Prediction:', y_pred.item())
