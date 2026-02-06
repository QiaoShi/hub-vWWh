import numpy as np
import matplotlib.pyplot as plt
import torch
from click.core import F


#生成数据
def generate_sin():
    x_numpy = np.linspace(0, 2*np.pi, 1000)
    y_numpy = np.sin(x_numpy) + np.random.normal(0, 0.01, 1000)
    # print(x_numpy, y_numpy)
    x_numpy = x_numpy.reshape(-1, 1)
    y_numpy = y_numpy.reshape(-1, 1)

    x_train = x_numpy[:800]
    x_test = x_numpy[800:]
    y_train = y_numpy[:800]
    y_test = y_numpy[800:]
    x_train_tensor = torch.Tensor(x_train)
    x_test_tensor = torch.Tensor(x_test)
    y_train_tensor = torch.Tensor(y_train)
    y_test_tensor = torch.Tensor(y_test)
    return {
        'x_train': x_train,
        'x_test': x_test,
        'y_train': y_train,
        'y_test': y_test,
        'x_train_tensor': x_train_tensor,
        'x_test_tensor': x_test_tensor,
        'y_train_tensor': y_train_tensor,
        'y_test_tensor': y_test_tensor
    }

class mymodel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(mymodel, self).__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim1),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim1, hidden_dim2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim2, hidden_dim1 // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim1 // 2, output_dim)
        )

        for m in self.network:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_normal_(m.weight)  # Xavier初始化
                torch.nn.init.constant_(m.bias, 0.0)


    def forward(self, x):
        return self.network(x)

def train_model(model, lr, n_epoch, batch_size, data):
    # 准备数据加载器
    train_dataset = torch.utils.data.TensorDataset(
        data['x_train_tensor'], data['y_train_tensor']
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    ln_loss = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr)  # 改用Adam优化器
    train_losses = []
    test_losses = []
    for epoch in range(n_epoch):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad()
            y_pre = model(batch_x)
            loss = ln_loss(y_pre, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches += 1
        print(f"Epoch: {epoch}/{n_epoch},epoch_loss={epoch_loss}")
        avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0
        train_losses.append(avg_train_loss)

        model.eval()
        test_x = data['x_test_tensor']
        test_y = data['y_test_tensor']
        y_test = model(test_x)
        y_loss = ln_loss(y_test, test_y)
        test_losses.append(y_loss)
    return train_losses, test_losses

def model_eval(model, data):
    # 确保输入维度正确
    x_train = data['x_train_tensor']
    x_test = data['x_test_tensor']

    # 预测
    model.eval()
    with torch.no_grad():
        y_pred_train = model(x_train)
        y_pred_test = model(x_test)

    # 转换为numpy
    y_pred_train_np = y_pred_train.detach().numpy().flatten()
    y_pred_test_np = y_pred_test.detach().numpy().flatten()

    return y_pred_train_np, y_pred_test_np
def main():
    # 生成sin数据
    data = generate_sin()

    #创建模型
    input_dim = 1
    hidden_dim1 = 128
    hidden_dim2 = 256
    output_dim = 1
    model = mymodel(input_dim, hidden_dim1, hidden_dim2, output_dim)

    # 训练模型
    learning_rate = 0.001
    n_epoch = 1000
    batch_size = 32
    train_loss, test_loss = train_model(model, learning_rate, n_epoch, batch_size, data)
    y_pred_train, y_pred_test = model_eval(model, data)

    # plt.figure(figsize=(10, 6))
    # x_all = np.concatenate([data['x_train'].flatten(), data['x_test'].flatten()])
    # y_pred_all = np.concatenate([y_pred_train, y_pred_test])
    # sort_idx = np.argsort(x_all)
    # x_sorted = x_all[sort_idx]
    # y_pred_sorted = y_pred_all[sort_idx]
    # plt.plot(x_sorted, y_pred_sorted, 'r-', linewidth=2,
    #          label='神经网络预测', alpha=0.8)
    # plt.show()

    # 绘图
    plt.figure(figsize=(12, 8))

    # 原始数据
    x_all = np.concatenate([data['x_train'].flatten(), data['x_test'].flatten()])
    y_all = np.concatenate([data['y_train'].flatten(), data['y_test'].flatten()])
    y_pred_all = np.concatenate([y_pred_train, y_pred_test])

    # 按x排序
    sort_idx = np.argsort(x_all)
    x_sorted = x_all[sort_idx]
    y_true_sorted = y_all[sort_idx]
    y_pred_sorted = y_pred_all[sort_idx]

    # 绘制真实数据和预测数据
    plt.scatter(x_all, y_all, s=10, alpha=0.6, label='原始数据（带噪声）', color='blue')
    plt.plot(x_sorted, y_pred_sorted, 'r-', linewidth=3, label='神经网络预测', alpha=0.8)

    # 绘制真实sin曲线对比
    x_true = np.linspace(0, 2 * np.pi, 1000)
    y_true = np.sin(x_true)
    plt.plot(x_true, y_true, 'g--', linewidth=2, label='真实sin曲线', alpha=0.7)

    plt.xlabel('x')
    plt.ylabel('sin(x)')
    plt.title('神经网络拟合正弦函数')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 绘制损失曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_loss, label='训练损失')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('训练损失曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(test_loss, label='测试损失')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('测试损失曲线')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    print(f"最终训练损失: {train_loss[-1]:.6f}")
    print(f"最终测试损失: {test_loss[-1]:.6f}")

    # 测试

if __name__ == "__main__":
    main()

