"""
深度学习文本分类程序
使用PyTorch实现基于字符级词袋模型（Char BoW）的文本分类
"""

import pandas as pd  # 用于读取和处理CSV数据文件
import torch  # PyTorch核心库，提供张量操作和自动求导功能
import torch.nn as nn  # PyTorch神经网络模块，包含各种层和激活函数
import torch.optim as optim  # PyTorch优化器模块，用于更新模型参数
from torch.utils.data import Dataset, DataLoader  # Dataset：自定义数据集基类；DataLoader：批量加载数据
import matplotlib.pyplot as plt  # 用于绘制图表，可视化loss变化

# ==================== 数据加载和预处理 ====================
# 读取CSV文件，sep='\t'表示使用制表符分隔，header=None表示没有表头
# 数据格式：第一列是文本，第二列是标签
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)

# 提取所有文本数据，转换为Python列表
# dataset[0]表示第一列（文本列），tolist()转换为列表
texts = dataset[0].tolist()

# 提取所有标签数据，转换为Python列表
# dataset[1]表示第二列（标签列），tolist()转换为列表
string_labels = dataset[1].tolist()

# ==================== 标签编码 ====================
# 创建标签到数字索引的映射字典
# set(string_labels)获取所有唯一的标签，enumerate为其分配索引0,1,2...
# 例如：{'Travel-Query': 0, 'Music-Play': 1, 'Video-Play': 2, ...}
label_to_index = {label: i for i, label in enumerate(set(string_labels))}

# 将字符串标签转换为数字标签（模型需要数字标签）
# 遍历每个原始标签，通过label_to_index字典查找对应的数字索引
numerical_labels = [label_to_index[label] for label in string_labels]

# ==================== 字符级词汇表构建 ====================
# 创建字符到索引的映射字典，用于将文本中的每个字符转换为数字
# '<pad>'是填充标记，索引为0，用于填充短文本
char_to_index = {'<pad>': 0}

# 遍历所有文本，构建完整的字符词汇表
for text in texts:
    # 遍历文本中的每个字符
    for char in text:
        # 如果字符不在字典中，为其分配新的索引
        # len(char_to_index)确保每个新字符获得递增的索引值
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

# 创建索引到字符的反向映射，用于后续可能的解码操作
# 例如：{0: '<pad>', 1: '还', 2: '有', ...}
index_to_char = {i: char for char, i in char_to_index.items()}

# 词汇表大小：所有唯一字符的数量（包括填充标记）
vocab_size = len(char_to_index)

# ==================== 超参数设置 ====================
# 最大文本长度：超过此长度的文本会被截断，不足的会用0填充
max_len = 40


# ==================== 自定义数据集类 ====================
class CharBoWDataset(Dataset):
    """
    字符级词袋模型数据集类
    将文本转换为词袋向量（Bag of Words），统计每个字符出现的次数
    """
    
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        """
        初始化数据集
        
        Args:
            texts: 文本列表
            labels: 数字标签列表
            char_to_index: 字符到索引的映射字典
            max_len: 最大文本长度
            vocab_size: 词汇表大小
        """
        # 保存原始文本数据
        self.texts = texts
        
        # 将标签转换为PyTorch张量，dtype=torch.long表示长整型（用于分类任务）
        self.labels = torch.tensor(labels, dtype=torch.long)
        
        # 保存字符映射字典
        self.char_to_index = char_to_index
        
        # 保存最大长度
        self.max_len = max_len
        
        # 保存词汇表大小
        self.vocab_size = vocab_size
        
        # 创建词袋向量（在初始化时一次性创建，避免重复计算）
        self.bow_vectors = self._create_bow_vectors()

    def _create_bow_vectors(self):
        """
        创建词袋向量
        词袋模型：统计每个字符在文本中出现的次数，忽略字符的顺序
        
        Returns:
            torch.Tensor: 形状为 [样本数, 词汇表大小] 的张量
        """
        # 第一步：将文本转换为字符索引序列
        tokenized_texts = []
        for text in self.texts:
            # 将文本中的每个字符转换为对应的索引
            # text[:self.max_len] 截断超过最大长度的文本
            # char_to_index.get(char, 0) 如果字符不在字典中，返回0（填充标记）
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            
            # 如果文本长度不足max_len，用0（填充标记）补齐
            # 例如：如果max_len=40，文本只有20个字符，则后面补20个0
            tokenized += [0] * (self.max_len - len(tokenized))
            
            # 将处理后的索引序列添加到列表
            tokenized_texts.append(tokenized)

        # 第二步：将字符索引序列转换为词袋向量
        bow_vectors = []
        for text_indices in tokenized_texts:
            # 创建一个全零向量，长度为词汇表大小
            # 这个向量将统计每个字符出现的次数
            bow_vector = torch.zeros(self.vocab_size)
            
            # 遍历文本中的每个字符索引
            for index in text_indices:
                # 如果索引不为0（不是填充标记），则在对应位置计数+1
                if index != 0:
                    bow_vector[index] += 1
            
            # 将当前文本的词袋向量添加到列表
            bow_vectors.append(bow_vector)
        
        # 将所有词袋向量堆叠成一个张量
        # 结果形状：[样本数, 词汇表大小]
        return torch.stack(bow_vectors)

    def __len__(self):
        """
        返回数据集的大小（样本数量）
        PyTorch的Dataset类需要实现此方法
        """
        return len(self.texts)

    def __getitem__(self, idx):
        """
        根据索引获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            tuple: (词袋向量, 标签)
        """
        # 返回指定索引的词袋向量和对应的标签
        return self.bow_vectors[idx], self.labels[idx]


# ==================== 神经网络模型定义 ====================
class SimpleClassifier(nn.Module):
    """
    简单的全连接神经网络分类器
    结构：输入层 -> 第一隐藏层 -> 第二隐藏层 -> 输出层
    两层隐藏层可以增加模型的表达能力，学习更复杂的特征模式
    """
    
    def __init__(self, input_dim, hidden_dim1,  hidden_dim2, output_dim):
        """
        初始化模型
        
        Args:
            input_dim: 输入维度（词汇表大小，即词袋向量的长度）
            hidden_dim1: 第一隐藏层维度（第一隐藏层神经元的数量）
            hidden_dim2: 第二隐藏层维度（第二隐藏层神经元的数量）
            output_dim: 输出维度（类别数量）
        """
        # 调用父类nn.Module的初始化方法
        super(SimpleClassifier, self).__init__()
        
        # 第一层全连接层（输入层到第一隐藏层）
        # nn.Linear(input_dim, hidden_dim1) 表示线性变换：y = xW^T + b
        # 输入维度：input_dim，输出维度：hidden_dim1
        # 注意：self.fc1 是一个独立的层对象，有自己的权重矩阵W1和偏置b1
        # W1的形状：[hidden_dim1, input_dim]，b1的形状：[hidden_dim1]
        self.fc1 = nn.Linear(input_dim, hidden_dim1)
        
        # ReLU激活函数：f(x) = max(0, x)
        # 引入非线性，使模型能够学习复杂的模式
        self.relu = nn.ReLU()
        
        # 第二层全连接层（第一隐藏层到第二隐藏层）
        # 输入维度：hidden_dim1，输出维度：hidden_dim2
        # 注意：self.fc2 是另一个独立的层对象，有自己的权重矩阵W2和偏置b2
        # W2的形状：[hidden_dim2, hidden_dim1]，b2的形状：[hidden_dim2]
        # 虽然可能hidden_dim1 == hidden_dim2，但fc1和fc2是完全不同的层！
        # 它们有独立的参数，在训练时会分别更新
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        
        # 第三层全连接层（第二隐藏层到输出层）
        # 输入维度：hidden_dim2，输出维度：output_dim（类别数）
        # 注意：self.fc3 也是独立的层对象，有自己的权重矩阵W3和偏置b3
        self.fc3 = nn.Linear(hidden_dim2, output_dim)

    def forward(self, x):
        """
        前向传播：定义数据如何通过网络
        
        Args:
            x: 输入张量，形状为 [batch_size, input_dim]
            
        Returns:
            输出张量，形状为 [batch_size, output_dim]
            注意：输出是原始logits（未经过softmax），CrossEntropyLoss内部会处理
        """
        # 第一层：线性变换
        # 输入x经过全连接层fc1，得到第一隐藏层输出
        # 计算过程：out1 = x @ W1^T + b1
        # 输入形状：[batch_size, input_dim]
        # 输出形状：[batch_size, hidden_dim1]
        # 注意：这里的out1是一个新的tensor，与输入x不同
        out = self.fc1(x)
        
        # 激活函数：ReLU非线性变换
        # 将负值置为0，保留正值，增加模型的非线性表达能力
        # 输出形状：[batch_size, hidden_dim1]（形状不变，只是值被激活函数处理）
        out = self.relu(out)
        
        # 第二层：线性变换（第一隐藏层到第二隐藏层）
        # 第一隐藏层的输出out1经过全连接层fc2，得到第二隐藏层输出
        # 计算过程：out2 = out1 @ W2^T + b2
        # 输入形状：[batch_size, hidden_dim1]
        # 输出形状：[batch_size, hidden_dim2]
        # 重要理解：
        # 1. out2是一个全新的tensor，与out1不同（虽然可能维度相同）
        # 2. fc2有自己独立的参数W2和b2，与fc1的参数W1和b1完全不同
        # 3. 即使hidden_dim1 == hidden_dim2，这两个层也是不同的！
        #    它们学习不同的特征表示，第一层学习基础特征，第二层学习更高级的特征组合
        out = self.fc2(out)
        
        # 激活函数：ReLU非线性变换
        # 再次应用ReLU激活函数，进一步增强模型的非线性表达能力
        # 输出形状：[batch_size, hidden_dim2]
        out = self.relu(out)
        
        # 第三层：线性变换（第二隐藏层到输出层）
        # 第二隐藏层的输出out2经过全连接层fc3，得到最终的logits（原始分数）
        # 计算过程：output = out2 @ W3^T + b3
        # 输入形状：[batch_size, hidden_dim2]
        # 输出形状：[batch_size, output_dim]
        out = self.fc3(out)
        
        # 返回输出（注意：这里没有softmax，因为CrossEntropyLoss内部会计算）
        return out


# ==================== 数据准备 ====================
# 创建数据集实例
# 传入文本、标签、字符映射、最大长度和词汇表大小
char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size)

# 创建数据加载器
# batch_size=32: 每批处理32个样本
# shuffle=True: 每个epoch开始时打乱数据顺序，提高训练效果
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)

# ==================== 模型初始化 ====================
# 第一隐藏层维度：128个神经元
# 这个值影响模型的容量，越大模型越复杂，但可能过拟合
hidden_dim1 = 128

# 第二隐藏层维度：可以设置与第一层相同或不同
# 如果设置为None，则使用hidden_dim1（两层维度相同）
# 如果设置为不同值，例如64，则第二层会更小（降维）或更大（升维）
# 常见做法：
# - 相同维度：hidden_dim2 = 128（保持维度）
# - 降维：hidden_dim2 = 64（压缩特征）
# - 升维：hidden_dim2 = 256（扩展特征）
hidden_dim2 = 64  

# 输出维度：类别数量（等于标签的种类数）
output_dim = len(label_to_index)

# 创建模型实例
# vocab_size: 输入维度（词袋向量的长度）
# hidden_dim1: 第一隐藏层维度
# hidden_dim2: 第二隐藏层维度（可以为None使用相同维度）
# output_dim: 输出维度（类别数）
# 
# 重要理解：
# - fc1和fc2是两个完全独立的层，即使维度相同，它们也有不同的参数
# - fc1的权重矩阵形状：[hidden_dim1, vocab_size]
# - fc2的权重矩阵形状：[hidden_dim2, hidden_dim1]
# - 这两个层在训练时会分别更新各自的参数
model = SimpleClassifier(vocab_size, hidden_dim1, output_dim, hidden_dim2)

# ==================== 损失函数和优化器 ====================
# 交叉熵损失函数
# 适用于多分类任务，内部会自动应用softmax和计算交叉熵
# 公式：Loss = -log(P(真实类别))
criterion = nn.CrossEntropyLoss()

# 随机梯度下降优化器
# model.parameters(): 获取模型所有可训练参数（权重和偏置）
# lr=0.01: 学习率，控制参数更新的步长
optimizer = optim.SGD(model.parameters(), lr=0.01)

# ==================== 训练过程 ====================
# epoch（轮次）：将整个数据集遍历一次
# batch（批次）：每次训练使用的一批样本
# 例如：12000个样本，batch_size=100，则一个epoch有12000/100=120个batch

# 训练轮数：模型将遍历整个数据集10次
num_epochs = 10

# 用于存储每个epoch的平均loss，用于后续绘制折线图
epoch_losses = []

# 外层循环：遍历每个epoch
for epoch in range(num_epochs):
    # 设置模型为训练模式
    # 这会启用dropout、batch normalization等训练时的行为
    model.train()
    
    # 累计损失值，用于计算平均损失
    running_loss = 0.0
    
    # 内层循环：遍历每个batch
    # enumerate(dataloader)返回(索引, (输入, 标签))
    for idx, (inputs, labels) in enumerate(dataloader):
        # 清零梯度
        # PyTorch会累积梯度，所以在每次反向传播前需要清零
        optimizer.zero_grad()
        
        # 前向传播：将输入数据传入模型，得到预测输出
        # inputs形状：[batch_size, vocab_size]
        # outputs形状：[batch_size, output_dim]
        outputs = model(inputs)
        
        # 计算损失：比较预测输出和真实标签
        # outputs: 模型预测的logits
        # labels: 真实的类别索引
        loss = criterion(outputs, labels)
        
        # 反向传播：计算损失对每个参数的梯度
        # 通过链式法则，从输出层向输入层传播梯度
        loss.backward()
        
        # 更新参数：根据梯度使用优化器更新模型参数
        # 参数更新公式：θ = θ - lr * ∇θ
        optimizer.step()
        
        # 累加损失值（loss.item()将张量转换为Python数值）
        running_loss += loss.item()
        
        # 每50个batch打印一次当前损失
        if idx % 50 == 0:
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

    # 每个epoch结束后，计算并保存平均损失
    # running_loss / len(dataloader) 计算该epoch的平均损失
    avg_loss = running_loss / len(dataloader)
    epoch_losses.append(avg_loss)  # 将平均loss添加到列表，用于后续绘图
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")


# ==================== 绘制Loss变化折线图函数 ====================
def plot_training_loss(epoch_losses, title='Training Loss Over Epochs', figsize=(10, 6)):
    """
    绘制训练过程中的Loss变化折线图
    
    Args:
        epoch_losses (list): 每个epoch的平均loss值列表
        title (str): 图表标题，默认为'Training Loss Over Epochs'
        figsize (tuple): 图表大小，默认为(10, 6)
    """
    # 检查是否有loss数据
    if not epoch_losses:
        print("警告：没有loss数据可绘制！")
        return
    
    # 创建图表，设置图表大小
    plt.figure(figsize=figsize)
    
    # 绘制折线图
    # epoch_losses是y轴数据（loss值）
    # range(1, len(epoch_losses) + 1)是x轴数据（epoch编号，从1开始）
    plt.plot(range(1, len(epoch_losses) + 1), epoch_losses, 
             marker='o', linewidth=2, markersize=8, label='Training Loss')
    
    # 设置图表标题和标签
    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    
    # 添加网格线，使图表更易读
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 设置x轴刻度为整数（每个epoch）
    plt.xticks(range(1, len(epoch_losses) + 1))
    
    # 添加图例
    plt.legend(loc='upper right')
    
    # 调整布局，确保标签不被裁剪
    plt.tight_layout()
    
    # 显示图表
    plt.show()
    
    # 打印最终loss信息
    print(f"\n训练完成！最终Loss: {epoch_losses[-1]:.4f}")
    print(f"Loss从 {epoch_losses[0]:.4f} 降低到 {epoch_losses[-1]:.4f}")


# ==================== 调用绘图函数 ====================
# 绘制训练过程中的Loss变化折线图
plot_training_loss(epoch_losses)


# ==================== 预测函数 ====================
def classify_text(text, model, char_to_index, vocab_size, max_len, index_to_label):
    """
    对单个文本进行分类预测
    
    Args:
        text: 待分类的文本字符串
        model: 训练好的模型
        char_to_index: 字符到索引的映射字典
        vocab_size: 词汇表大小
        max_len: 最大文本长度
        index_to_label: 索引到标签的映射字典
        
    Returns:
        预测的类别标签（字符串）
    """
    # 第一步：将文本转换为字符索引序列
    # 截断超过最大长度的文本，并将每个字符转换为索引
    tokenized = [char_to_index.get(char, 0) for char in text[:max_len]]
    
    # 如果文本长度不足，用0（填充标记）补齐
    tokenized += [0] * (max_len - len(tokenized))

    # 第二步：创建词袋向量
    # 初始化一个全零向量
    bow_vector = torch.zeros(vocab_size)
    
    # 统计每个字符出现的次数
    for index in tokenized:
        # 如果索引不为0（不是填充标记），计数+1
        if index != 0:
            bow_vector[index] += 1

    # 第三步：添加batch维度
    # unsqueeze(0)在维度0添加一个维度
    # 从 [vocab_size] 变为 [1, vocab_size]，因为模型期望batch维度
    bow_vector = bow_vector.unsqueeze(0)

    # 第四步：模型预测
    # 设置模型为评估模式（禁用dropout等训练时的行为）
    model.eval()
    
    # 禁用梯度计算，节省内存和计算资源（预测时不需要梯度）
    with torch.no_grad():
        # 前向传播，得到输出logits
        output = model(bow_vector)

    # 第五步：获取预测结果
    # torch.max(output, 1) 在维度1（类别维度）上找最大值
    # 返回 (最大值, 最大值的索引)
    # _ 表示忽略最大值，只取索引（即预测的类别索引）
    _, predicted_index = torch.max(output, 1)
    
    # 将张量转换为Python整数
    predicted_index = predicted_index.item()
    
    # 通过索引到标签的映射，将数字索引转换为字符串标签
    predicted_label = index_to_label[predicted_index]

    # 返回预测的类别标签
    return predicted_label


# ==================== 模型预测示例 ====================
# 创建索引到标签的反向映射字典
# 用于将模型输出的数字索引转换回原始标签字符串
index_to_label = {i: label for label, i in label_to_index.items()}

# 测试文本1
new_text = "帮我导航到北京"
# 调用预测函数进行分类
predicted_class = classify_text(new_text, model, char_to_index, vocab_size, max_len, index_to_label)
# 打印预测结果
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

# 测试文本2
new_text_2 = "查询明天北京的天气"
# 调用预测函数进行分类
predicted_class_2 = classify_text(new_text_2, model, char_to_index, vocab_size, max_len, index_to_label)
# 打印预测结果
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")
