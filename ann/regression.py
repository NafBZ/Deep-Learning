import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class RegressionModel(nn.Module):
    def __init__(self):
        super(RegressionModel, self).__init__()
        self.regression = nn.Sequential(
            nn.Linear(1, 1),  # input layer
            nn.ReLU(),       # activation function
            nn.Linear(1, 1)   # output layer
        )

    def forward(self, x):
        return self.regression(x)

class RegressionTrainer:
    def __init__(self, model, learning_rate=0.01, epochs=50):
        self.model = model
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.loss_function = nn.MSELoss()
        self.optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    def train(self, x, y):
        losses = torch.zeros(self.epochs)

        for epoch in range(self.epochs):
            # forward pass
            pred = self.model(x)

            # loss calculation
            loss_value = self.loss_function(pred, y)
            losses[epoch] = loss_value.item()

            # backpropagation
            self.optimizer.zero_grad()
            loss_value.backward()
            self.optimizer.step()

        predictions = self.model(x).detach().numpy()

        return predictions, losses

def data(m):
    N = 100
    x = torch.randn(N, 1)
    y = m * x + torch.randn(N, 1) / 2
    return x, y

# Create an instance of the model and trainer
model = RegressionModel()
trainer = RegressionTrainer(model)

# Generate data
x, y = data(0.7)

# Train the model
preds, losses = trainer.train(x, y)

# Plot results
fig, ax = plt.subplots(1, 2, figsize=(12, 5))
ax[0].plot(losses, 'o', lw=0.1, markerfacecolor='w')
ax[0].set_xlabel('Epochs')
ax[0].set_ylabel('Loss')

ax[1].plot(x, y, 'o', label='Actual Data')
ax[1].plot(x, preds, 'rs', label='Predictions')
ax[1].set_xlabel('x')
ax[1].set_ylabel('y')

ax[1].legend()
plt.show()
