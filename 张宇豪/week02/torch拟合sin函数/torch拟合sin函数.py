import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

num = 500
X_numpy = np.sort(np.random.rand(num, 1) * 10 - 5, axis=0)
Y_numpy = np.sin(X_numpy) + np.random.rand(num, 1) * 0.1

X = torch.from_numpy(X_numpy).float()
Y = torch.from_numpy(Y_numpy).float()

class SinModel(nn.Module):
    def __init__(self, input_size, hidden1_size, hidden2_size, output_size):
        super(SinModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden1_size),
            nn.ReLU(),
            nn.Linear(hidden1_size, hidden2_size),
            nn.ReLU(),
            nn.Linear(hidden2_size, output_size)
        )
    def forward(self, X):
        return self.network(X)

input_size, hidden1_size, hidden2_size, output_size = 1, 64, 64, 1
model = SinModel(input_size, hidden1_size, hidden2_size, output_size)
loss_fn = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

num_epochs = 1000
for epoch in range(num_epochs):
    model.train()
    y_pred = model(X)
    loss = loss_fn(y_pred, Y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch: {epoch + 1} / {num_epochs}, Loss: {loss.item():.4f}')

print('训练完成！')

model.eval()
with torch.no_grad():
    y_predicted = model(X)

# 绘图
plt.figure(figsize=(10, 6))
plt.scatter(X_numpy, Y_numpy, label='Raw data', color='blue', alpha=0.6)
plt.plot(X_numpy, y_predicted, label='Predicted data', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
