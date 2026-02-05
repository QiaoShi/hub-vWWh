"""
使用多层神经网络拟合sin函数
这是一个回归任务，展示神经网络如何学习非线性函数
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader


# ==================== 生成sin函数数据 ====================
def generate_sin_data(x_range=(-2*np.pi, 2*np.pi), num_samples=1000):
    """
    生成sin函数的数据点
    
    Args:
        x_range: x的取值范围，默认为(-2π, 2π)
        num_samples: 生成的样本数量
        
    Returns:
        tuple: (x值数组, y值数组)
    """
    # 在指定范围内均匀采样x值
    x = np.linspace(x_range[0], x_range[1], num_samples)
    
    # 计算对应的sin函数值
    y = np.sin(x)
    
    return x, y


# ==================== 自定义数据集类 ====================
class SinDataset(Dataset):
    """
    sin函数数据集类
    用于PyTorch的DataLoader
    """
    
    def __init__(self, x, y):
        """
        初始化数据集
        
        Args:
            x: 输入x值（numpy数组）
            y: 输出y值（numpy数组，sin(x)的值）
        """
        # 将numpy数组转换为PyTorch张量，并转换为float32类型
        self.x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)  # 添加维度：[N] -> [N, 1]
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)  # 添加维度：[N] -> [N, 1]
    
    def __len__(self):
        """返回数据集大小"""
        return len(self.x)
    
    def __getitem__(self, idx):
        """根据索引获取单个样本"""
        return self.x[idx], self.y[idx]


# ==================== 神经网络模型定义 ====================
class SinRegressor(nn.Module):
    """
    多层神经网络回归器
    用于拟合sin函数
    结构：输入层 -> 隐藏层1 -> 隐藏层2 -> 隐藏层3 -> 输出层
    """
    
    def __init__(self, input_dim=1, hidden_dims=[64, 128, 64], output_dim=1):
        """
        初始化模型
        
        Args:
            input_dim: 输入维度（对于sin函数，输入是单个x值，所以是1）
            hidden_dims: 各隐藏层的维度列表，例如[64, 128, 64]表示三层隐藏层
            output_dim: 输出维度（对于回归任务，输出是单个y值，所以是1）
        """
        super(SinRegressor, self).__init__()
        
        # 构建多层网络
        layers = []
        
        # 第一层：输入层到第一隐藏层
        layers.append(nn.Linear(input_dim, hidden_dims[0]))
        layers.append(nn.ReLU())
        
        # 中间隐藏层：依次连接
        for i in range(len(hidden_dims) - 1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            layers.append(nn.ReLU())
        
        # 最后一层：最后一层隐藏层到输出层
        layers.append(nn.Linear(hidden_dims[-1], output_dim))
        
        # 使用Sequential将所有层组合在一起
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入张量，形状为 [batch_size, 1]
            
        Returns:
            输出张量，形状为 [batch_size, 1]
        """
        return self.network(x)


# ==================== 训练函数 ====================
def train_model(model, dataloader, num_epochs=1000, learning_rate=0.001):
    """
    训练模型
    
    Args:
        model: 神经网络模型
        dataloader: 数据加载器
        num_epochs: 训练轮数
        learning_rate: 学习率
        
    Returns:
        list: 每个epoch的损失值列表
    """
    # 使用均方误差损失函数（MSE Loss）
    # 适用于回归任务，计算预测值与真实值的平方差
    criterion = nn.MSELoss()
    
    # 使用Adam优化器，比SGD更稳定，收敛更快
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 存储每个epoch的损失值
    losses = []
    
    print("开始训练...")
    for epoch in range(num_epochs):
        # 设置模型为训练模式
        model.train()
        
        # 累计损失
        epoch_loss = 0.0
        
        # 遍历每个batch
        for batch_x, batch_y in dataloader:
            # 清零梯度
            optimizer.zero_grad()
            
            # 前向传播：得到预测值
            predictions = model(batch_x)
            
            # 计算损失
            loss = criterion(predictions, batch_y)
            
            # 反向传播
            loss.backward()
            
            # 更新参数
            optimizer.step()
            
            # 累加损失
            epoch_loss += loss.item()
        
        # 计算平均损失
        avg_loss = epoch_loss / len(dataloader)
        losses.append(avg_loss)
        
        # 每100个epoch打印一次
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.6f}")
    
    print("训练完成！")
    return losses


# ==================== 可视化函数 ====================
def visualize_results(model, x_train, y_train, x_test=None, y_test=None, losses=None):
    """
    可视化拟合结果
    
    Args:
        model: 训练好的模型
        x_train: 训练数据的x值
        y_train: 训练数据的y值（真实sin值）
        x_test: 测试数据的x值（可选）
        y_test: 测试数据的y值（可选）
        losses: 训练损失列表（可选）
    """
    # 设置模型为评估模式
    model.eval()
    
    # 创建图形，包含多个子图
    if losses is not None:
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    else:
        fig, axes = plt.subplots(1, 1, figsize=(10, 6))
        axes = [axes]
    
    # ========== 子图1：拟合结果对比 ==========
    ax1 = axes[0]
    
    # 绘制真实的sin函数曲线（训练数据）
    ax1.plot(x_train, y_train, 'b-', label='True sin(x)', linewidth=2, alpha=0.7)
    
    # 使用模型进行预测
    with torch.no_grad():  # 禁用梯度计算，节省内存
        # 将x_train转换为张量
        x_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
        # 预测
        y_pred = model(x_tensor).squeeze(1).numpy()
    
    # 绘制模型预测的曲线
    ax1.plot(x_train, y_pred, 'r--', label='Predicted', linewidth=2, alpha=0.8)
    
    # 如果有测试数据，也绘制出来
    if x_test is not None and y_test is not None:
        ax1.plot(x_test, y_test, 'g-', label='Test sin(x)', linewidth=1, alpha=0.5)
    
    # 设置图表属性
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y = sin(x)', fontsize=12)
    ax1.set_title('Sin Function Fitting', fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # ========== 子图2：训练损失曲线 ==========
    if losses is not None:
        ax2 = axes[1]
        ax2.plot(losses, 'b-', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Loss (MSE)', fontsize=12)
        ax2.set_title('Training Loss', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')  # 使用对数刻度，更好地显示损失下降
    
    plt.tight_layout()
    plt.show()


# ==================== 主函数 ====================
def main():
    """
    主函数：执行完整的训练和可视化流程
    """
    # 设置随机种子，确保结果可复现
    torch.manual_seed(42)
    np.random.seed(42)
    
    # 1. 生成训练数据
    print("生成训练数据...")
    x_train, y_train = generate_sin_data(x_range=(-2*np.pi, 2*np.pi), num_samples=1000)
    
    # 2. 创建数据集和数据加载器
    dataset = SinDataset(x_train, y_train)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # 3. 创建模型
    # 定义隐藏层结构：3层隐藏层，维度分别为64, 128, 64
    # 这种"宽-窄"的结构可以帮助模型学习复杂的特征
    hidden_dims = [64, 128, 64]
    model = SinRegressor(input_dim=1, hidden_dims=hidden_dims, output_dim=1)
    
    print(f"模型结构：")
    print(model)
    print(f"\n模型参数数量：{sum(p.numel() for p in model.parameters())}")
    
    # 4. 训练模型
    losses = train_model(model, dataloader, num_epochs=1000, learning_rate=0.001)
    
    # 5. 生成测试数据（在更大的范围内测试模型的泛化能力）
    print("\n生成测试数据...")
    x_test, y_test = generate_sin_data(x_range=(-3*np.pi, 3*np.pi), num_samples=1500)
    
    # 6. 可视化结果
    print("\n绘制结果...")
    visualize_results(model, x_train, y_train, x_test, y_test, losses)
    
    # 7. 计算并打印评估指标
    model.eval()
    with torch.no_grad():
        # 在训练数据上计算误差
        x_train_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1)
        y_pred_train = model(x_train_tensor).squeeze(1).numpy()
        train_mse = np.mean((y_pred_train - y_train) ** 2)
        train_mae = np.mean(np.abs(y_pred_train - y_train))
        
        # 在测试数据上计算误差
        x_test_tensor = torch.tensor(x_test, dtype=torch.float32).unsqueeze(1)
        y_pred_test = model(x_test_tensor).squeeze(1).numpy()
        test_mse = np.mean((y_pred_test - y_test) ** 2)
        test_mae = np.mean(np.abs(y_pred_test - y_test))
    
    print("\n" + "="*50)
    print("评估结果：")
    print(f"训练集 - MSE: {train_mse:.6f}, MAE: {train_mae:.6f}")
    print(f"测试集 - MSE: {test_mse:.6f}, MAE: {test_mae:.6f}")
    print("="*50)


if __name__ == '__main__':
    main()

