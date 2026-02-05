import torch
import numpy as np # cpu 环境（非深度学习中）下的矩阵运算、向量运算
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# 1. 生成模拟数据 (与之前相同)
X_numpy = np.random.rand(100, 1) * 2*np.pi
# 形状为 (100, 1) 的二维数组，其中包含 100 个在 [0, 1) 范围内均匀分布的随机浮点数。
y_numpy = 2*np.sin(X_numpy) + 2 + np.random.randn(100, 1)*0.2

X = torch.from_numpy(X_numpy).float() # torch 中 所有的计算 通过tensor 计算
y = torch.from_numpy(y_numpy).float()

print("数据生成完成。")
print("---" * 10)

# 3.定义多层神经网络模型
class SinTrainingModel(nn.Module):
    def __init__(self,input_dim,hidden1_dim,hidden2_dim,output_dim):
        super().__init__()
        self.fc1=nn.Linear(input_dim,hidden1_dim)
        self.Tanh1=nn.Tanh()
        self.fc2=nn.Linear(hidden1_dim,hidden2_dim)
        self.Tanh2=nn.Tanh()
        self.fc3=nn.Linear(hidden2_dim,output_dim)

    def forward(self,x):
        output = self.fc1(x)
        output = self.Tanh1(output)
        output = self.fc2(output)
        output = self.Tanh2(output)
        output = self.fc3(output)

        # 分离参数a和b
        a = output[:, 0:1]  # 第一个输出作为振幅a
        b = output[:, 1:2]  # 第二个输出作为偏移b

        # 计算 y = a·sin(x) + b
        y_pred = a * torch.sin(x) + b
        return y_pred, a, b  # 返回预测值和参数


input_dim= 1
hidden1_dim = 128
hidden2_dim = 64
output_dim = 2

# 初始化模型
model=SinTrainingModel(input_dim, hidden1_dim, hidden2_dim, output_dim)
loss_fn = torch.nn.MSELoss()
optimizer=optim.Adam(model.parameters(), lr=0.01)

print("初始化模型完成")
print("----"*10)


# 4. 训练模型
num_epochs = 100
for epoch in range(num_epochs):
    # 前向传播：手动计算 y_pred = a * sin(X) + b
    y_pred, a_pred, b_pred = model(X)
    # 计算损失
    loss = loss_fn(y_pred, y)

    # 反向传播和优化
    optimizer.zero_grad()  # 清空梯度， torch 梯度 累加
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数

    # 每100个 epoch 打印一次损失
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# 5. 打印最终学到的参数
print("\n训练完成！")


# 6. 绘制结果
x_plot = np.linspace(0, 2*np.pi, 200).reshape(-1, 1)
x_plot_tensor = torch.from_numpy(x_plot).float()
with torch.no_grad():
    y_final_pred, a_final, b_final = model(x_plot_tensor)
    learned_a = a_final.mean().item()
    learned_b = b_final.mean().item()
    print(f"拟合的斜率 a: {learned_a:.4f}")
    print(f"拟合的截距 b: {learned_b:.4f}")
    print("---" * 10)

plt.figure(figsize=(10, 6))
y_plot_data = y_final_pred.numpy()
plt.scatter(X_numpy, y_numpy, label='Raw data', color='blue', alpha=0.6)
plt.plot(x_plot, y_plot_data,label=f'Model: y = {learned_a:.2f}sin(x) + {learned_b:.2f}', color='red', linewidth=2)
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()
