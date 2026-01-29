import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# 1. 生成sin函数数据
X_numpy = np.random.uniform(-2 * np.pi, 2 * np.pi, (1000, 1))  # 生成更多数据
y_numpy = np.sin(X_numpy) + np.random.normal(0, 0.1, X_numpy.shape)  # 添加一些噪声

# 划分训练集和测试集
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    X_numpy, y_numpy, test_size=0.2, random_state=42
)

# 转换为PyTorch张量
X_train = torch.from_numpy(X_train_np).float()
y_train = torch.from_numpy(y_train_np).float()
X_test = torch.from_numpy(X_test_np).float()
y_test = torch.from_numpy(y_test_np).float()

print("数据生成完成。")
print(f"训练集大小: {X_train.shape[0]}, 测试集大小: {X_test.shape[0]}")
print("---" * 10)


# 2. 定义多层神经网络模型
class SinNet(torch.nn.Module):
    def __init__(self, input_size=1, hidden_layers=[64, 32, 16], output_size=1):
        """
        多层神经网络
        参数:
            input_size: 输入特征维度
            hidden_layers: 隐藏层神经元数量列表
            output_size: 输出维度
        """
        super(SinNet, self).__init__()

        # 构建网络层
        layers = []

        # 输入层到第一个隐藏层
        layers.append(torch.nn.Linear(input_size, hidden_layers[0]))
        layers.append(torch.nn.ReLU())

        # 添加隐藏层
        for i in range(len(hidden_layers) - 1):
            layers.append(torch.nn.Linear(hidden_layers[i], hidden_layers[i + 1]))
            layers.append(torch.nn.ReLU())

        # 最后一个隐藏层到输出层
        layers.append(torch.nn.Linear(hidden_layers[-1], output_size))

        # 组合所有层
        self.network = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


# 3. 初始化模型、损失函数和优化器
model = SinNet(input_size=1, hidden_layers=[128, 64, 32, 16], output_size=1)
print(f"模型结构:\n{model}")
print("---" * 10)

# 计算模型参数数量
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"总参数数量: {total_params}")
print(f"可训练参数数量: {trainable_params}")
print("---" * 10)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)  # 使用Adam优化器
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.8)  # 学习率调度器

# 4. 训练模型
num_epochs = 500
train_losses = []
test_losses = []

print("开始训练...")
for epoch in range(num_epochs):
    # 训练阶段
    model.train()
    y_pred_train = model(X_train)
    loss_train = loss_fn(y_pred_train, y_train)

    optimizer.zero_grad()
    loss_train.backward()
    optimizer.step()
    scheduler.step()

    # 测试阶段
    model.eval()
    with torch.no_grad():
        y_pred_test = model(X_test)
        loss_test = loss_fn(y_pred_test, y_test)

    # 记录损失
    train_losses.append(loss_train.item())
    test_losses.append(loss_test.item())

    # 每100个epoch打印一次
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], '
              f'Train Loss: {loss_train.item():.6f}, '
              f'Test Loss: {loss_test.item():.6f}, '
              f'LR: {scheduler.get_last_lr()[0]:.6f}')

print("\n训练完成！")
print("---" * 10)

# 5. 可视化训练过程
plt.figure(figsize=(12, 4))

# 损失曲线
plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Train Loss', alpha=0.8)
plt.plot(test_losses, label='Test Loss', alpha=0.8)
plt.xlabel('Epoch')
plt.ylabel('Loss (MSE)')
plt.title('Training and Test Loss')
plt.legend()
plt.grid(True, alpha=0.3)

# 最终预测结果
plt.subplot(1, 2, 2)

# 生成用于可视化的平滑数据
X_vis = np.linspace(-2 * np.pi, 2 * np.pi, 500).reshape(-1, 1)
y_sin_true = np.sin(X_vis)

# 使用模型预测
model.eval()
with torch.no_grad():
    X_vis_tensor = torch.from_numpy(X_vis).float()
    y_pred_vis = model(X_vis_tensor).numpy()

# 绘制原始sin函数
plt.plot(X_vis, y_sin_true, 'b-', label='True sin(x)', linewidth=2, alpha=0.7)

# 绘制训练数据点
plt.scatter(X_train_np, y_train_np, label='Training Data',
            color='green', alpha=0.3, s=10)

# 绘制测试数据点
plt.scatter(X_test_np, y_test_np, label='Test Data',
            color='red', alpha=0.5, s=15, marker='x')

# 绘制模型预测
plt.plot(X_vis, y_pred_vis, 'r-', label='Model Prediction', linewidth=2)

plt.xlabel('x')
plt.ylabel('y')
plt.title('sin(x) Function Fitting')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# 6. 在测试集上计算最终性能
model.eval()
with torch.no_grad():
    y_pred_final = model(X_test)
    final_test_loss = loss_fn(y_pred_final, y_test)
    # 计算R²分数
    ss_total = torch.sum((y_test - torch.mean(y_test)) ** 2)
    ss_residual = torch.sum((y_test - y_pred_final) ** 2)
    r2_score = 1 - (ss_residual / ss_total)

print(f"测试集最终性能:")
print(f"  测试损失 (MSE): {final_test_loss.item():.6f}")
print(f"  R²分数: {r2_score.item():.4f}")
print("---" * 10)

# 7. 添加额外的可视化：模型在局部区域的拟合效果
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 定义要查看的不同区域
regions = [
    (-2 * np.pi, 2 * np.pi, "Full Range"),
    (0, 2 * np.pi, "0 to 2π"),
    (-np.pi, np.pi, "-π to π"),
    (-1, 1, "Around Zero")
]

for i, (start, end, title) in enumerate(regions):
    ax = axes[i // 2, i % 2]

    # 生成该区域的数据
    X_region = np.linspace(start, end, 200).reshape(-1, 1)
    y_true_region = np.sin(X_region)

    # 模型预测
    with torch.no_grad():
        X_region_tensor = torch.from_numpy(X_region).float()
        y_pred_region = model(X_region_tensor).numpy()

    # 绘制
    ax.plot(X_region, y_true_region, 'b-', label='True sin(x)', linewidth=2, alpha=0.7)
    ax.plot(X_region, y_pred_region, 'r-', label='Model Prediction', linewidth=2)

    # 添加该区域的训练数据点
    mask = (X_train_np >= start) & (X_train_np <= end)
    if np.any(mask):
        ax.scatter(X_train_np[mask], y_train_np[mask],
                   color='green', alpha=0.3, s=10, label='Training Data')

    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_title(f'sin(x) Fitting - {title}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 8. 模型参数分布可视化（可选）
plt.figure(figsize=(14, 4))

for i, (name, param) in enumerate(model.named_parameters()):
    if 'weight' in name:
        plt.subplot(1, 3, i % 3 + 1)
        plt.hist(param.data.flatten().numpy(), bins=50, alpha=0.7, edgecolor='black')
        plt.xlabel('Weight Value')
        plt.ylabel('Frequency')
        plt.title(f'Weight Distribution: {name}')
        plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
