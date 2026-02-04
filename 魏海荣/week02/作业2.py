import torch
import numpy as np # cpu 环境（非深度学习中）下的矩阵运算、向量运算
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1. 生成模拟数据使用等间距的随机数【0， 2 pi】，np.linspace(start, end, num)
X_numpy = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1) # reshape(-1, 1) 将一维数组转换为二维数组，-1 表示自动计算行数，1 表示1列
# 形状为 (100, 1) 的二维数组，其中包含 100 个在 [0, 1) 范围内均匀分布的随机浮点数。

y_numpy = np.sin(X_numpy) + 0.1 * np.random.randn(100, 1)  # 添加一些噪声
X = torch.from_numpy(X_numpy).float() # torch 中 所有的计算 通过tensor 计算
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)


# 创建多层网络结构
class SinFittingMLP(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SinFittingMLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)  # 输入维度1，输出维度1
        self.tanh = torch.nn.Tanh() # 激活函数,Tanh
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)  # 输入维度1，输出维度1

    def forward(self, x):
        out = self.fc1(x)
        out = self.tanh(out)
        out = self.fc2(out)
        return out

# 2. 直接创建参数张量 a 和 b
# torch.randn() 生成随机值作为初始值。
# y = a * x + b
# requires_grad=True 是关键！它告诉 PyTorch 我们需要计算这些张量的梯度。
# a = torch.randn(1, requires_grad=True, dtype=torch.float)
# b = torch.randn(1, requires_grad=True, dtype=torch.float)

# print(f"初始参数 a: {a.item():.4f}")
# print(f"初始参数 b: {b.item():.4f}")
# print("---" * 10)

hidden_dim = 64
model = SinFittingMLP(input_dim=1, hidden_dim=hidden_dim, output_dim=1)

# 3. 定义损失函数和优化器
# 损失函数仍然是均方误差 (MSE)。
loss_fn = torch.nn.MSELoss() # 回归任务
# PyTorch 会自动根据这些参数的梯度来更新它们。
optimizer = optim.Adam(model.parameters(), lr=0.01)

# 4. 训练模型
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    # 前向传播：计算预测值
    y_pred = model(X)

    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度， torch 梯度 累加
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数

    # 每100个 epoch 打印一次损失
    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 打印最终学到的参数
print("\n训练完成！")
with torch.no_grad():  # 关闭梯度计算
    y_predicted = model(X)
    # 把tensor转换成numpy数组（方便和X_numpy/y_numpy一起绘图）
    y_predicted_numpy = y_predicted.numpy()


# 6. 绘制结果
plt.figure(figsize=(10, 6))
# 绘制原始数据点（带噪声的sin值）
plt.scatter(X_numpy, y_numpy, label='Raw data (sin + noise)', color='blue', alpha=0.6)
# 绘制模型拟合的曲线（用X_numpy和y_predicted_numpy）
# ******** 请你补全这行：plot的y轴用y_predicted_numpy ********
plt.plot(X_numpy, y_predicted_numpy, label='Model Fitting', color='red', linewidth=2)
plt.xlabel('X (0 ~ 2π)')
plt.ylabel('y (sin(X))')
plt.title('MLP Fitting Sin Function')
plt.legend()
plt.grid(True)
plt.show()
