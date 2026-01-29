import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
import torch
import numpy as np  # cpu 环境（非深度学习中）下的矩阵运算、向量运算
import torch.nn as nn
import matplotlib.pyplot as plt

# 1. 生成sin函数数据
X_numpy = np.linspace(-3, 3, 100).reshape(-1, 1)  # 生成100个-3-3之间的随机数

y_numpy = np.sin(X_numpy)  # 改为sin
X = torch.from_numpy(X_numpy).float()  # torch 中 所有的计算 通过tensor 计算
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)


# 2. 创建网络（原代码是手动参数，现在用网络）
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1, 20)  # 修改：1维输入，20个神经元
        self.fc2 = nn.Linear(20, 1)  # 修改：20 -> 1输出

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # 添加激活函数
        return self.fc2(x)


# 3. 定义损失函数和优化器
# 损失函数仍然是均方误差 (MSE)。
model = SimpleNet()
loss_fn = nn.MSELoss()  # 回归任务

# PyTorch 会自动根据这些参数的梯度来更新它们。
optimizer = torch.optim.Adam(model.parameters(), lr=0.01) # 优化器

# 4. 训练模型
num_epochs = 1000
for epoch in range(num_epochs):
    # 前向传播：手动计算 y_pred = sin(x)
    y_pred = model(X)

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度， torch 梯度 累加
    loss.backward()  # 计算梯度
    optimizer.step()  # 更新参数

    # 每100个 epoch 打印一次损失
    if epoch % 50 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 打印最终学到的参数
print("\n训练完成！")

# 6. 绘制结果
# 使用最终学到的参数 a 和 b 来计算拟合直线的 y 值
with torch.no_grad():
    y_predicted = model(X).numpy()

plt.scatter(X, y, label='True')
plt.plot(X, y_predicted, label='Predicted', color='red')
plt.legend()
plt.show()
