#理解rnn、lstm、gru的计算过程（面试用途），阅读官方文档 ：https://docs.pytorch.org/docs/2.4/nn.html#recurrent-layers
# 最终 使用 GRU 代替 LSTM 实现05_LSTM文本分类.py；05_LSTM文本分类.py
# 使用rnn/ lstm / gru 分别代替原始lstm，进行实验，对比精度

# rnn、lstm、gru 用于时间序列
#=================rnn=============================================================================
#核心是用当前输入 + 上一时刻隐藏状态，计算当前隐藏状态，解决序列数据的时序依赖问题
#但存在梯度消失 / 爆炸，无法捕捉长距离依赖。无记忆机制，长序列中梯度易消失，无法记住早期关键信息。
#=================lstm=============================================================================
#对 RNN 的隐藏层做了结构升级，引入细胞状态（Cell State） 和 3 个门控（输入门、遗忘门、输出门），
#通过门控控制信息的 “保留 / 遗忘 / 输出”，解决长距离依赖问题，是最经典的循环层。
#优点：解决长距离依赖；缺点：结构复杂，参数多，计算速度慢。
#=================gru=============================================================================
#LSTM 的简化版（2014 年提出），合并了细胞状态和隐藏状态，去掉输出门，仅保留重置门和更新门，
# 在保证大部分性能的前提下，减少参数、提升计算速度，是工业界更常用的选择。
#用更新门替代 LSTM 的遗忘门 + 输入门，用重置门控制是否遗忘上一时刻隐藏状态，结构更简洁；
#参数比 LSTM 少 40% 左右，计算快，长序列表现接近 LSTM；缺点：极长序列下性能略逊于 LSTM。

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()
# 标签向量化
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

# max length 最大输入的文本长度
max_len = 40

# 自定义数据集 - 》 为每个任务定义单独的数据集的读取方式，这个任务的输入和输出
# 统一的写法，底层pytorch 深度学习 / 大模型
class CharLSTMDataset(Dataset):
    # 初始化
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts # 文本输入
        self.labels = torch.tensor(labels, dtype=torch.long) # 文本对应的标签
        self.char_to_index = char_to_index # 字符到索引的映射关系
        self.max_len = max_len # 文本最大输入长度

    # 返回数据集样本个数
    def __len__(self):
        return len(self.texts)

    # 获取当个样本
    def __getitem__(self, idx):
        text = self.texts[idx]
        # pad and crop 取长补短
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]

# a = CharLSTMDataset()
# len(a) -> a.__len__
# a[0] -> a.__getitem__


# --- NEW LSTM Model Class ---


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,model_type = 1):
        #model_type:1:lstm,2:rnn,3:gru
        super(LSTMClassifier, self).__init__()
        self.model_type = model_type
        # 词表大小 转换后维度的维度
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # 随机编码的过程， 可训练的
        if model_type == 1:
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)  # 循环层
        elif model_type == 2:
            self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)  # 循环层
        elif model_type == 3:
            self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)  # 循环层
        else:
            print('没有该模型')

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # batch size * seq length -》 batch size * seq length * embedding_dim
        embedded = self.embedding(x)

        # batch size * seq length * embedding_dim -》 batch size * seq length * hidden_dim
        if self.model_type == 1:
            # LSTM输出：(所有时刻隐藏状态, (最后时刻隐藏状态, 细胞状态))
            rnn_out, (hidden_state, cell_state) = self.rnn(embedded)
        else:
            # RNN/GRU输出：(所有时刻隐藏状态, 最后时刻隐藏状态)
            rnn_out, hidden_state = self.rnn(embedded)

        # batch size * output_dim
        out = self.fc(hidden_state.squeeze(0))
        return out

def train_and_evaluate(model_type, dataloader, vocab_size, embedding_dim, hidden_dim, output_dim, num_epochs=4):
    """
    训练指定模型，并返回训练过程的损失和准确率
    :param model_type: 模型类型 rnn/lstm/gru
    :return: (model, epoch_losses, epoch_accs) 训练好的模型、各epoch损失、各epoch准确率
    """
    # 初始化模型、损失函数、优化器（与原始代码一致）
    model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim,model_type)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epoch_losses = []  # 保存各epoch的平均损失
    epoch_accs = []    # 保存各epoch的平均准确率

    print(f"\n==================== 开始训练 {'LSTM' if model_type == 1 else 'RNN' if model_type == 2 else 'GRU'} 模型 ====================")
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for idx, (inputs, labels) in enumerate(dataloader):
            # 梯度清零
            optimizer.zero_grad()
            # 前向传播
            outputs = model(inputs)
            # 计算损失
            loss = criterion(outputs, labels)
            # 反向传播+优化
            loss.backward()
            optimizer.step()

            # 统计损失和准确率
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # 打印批次信息
            if idx % 50 == 0:
                batch_acc = (predicted == labels).sum().item() / labels.size(0)
                print(f"Epoch {epoch+1}/{num_epochs} | Batch {idx} | Loss: {loss.item():.4f} | Batch Acc: {batch_acc:.4f}")

        # 计算当前epoch的平均损失和准确率
        avg_loss = running_loss / len(dataloader)
        avg_acc = correct / total
        epoch_losses.append(avg_loss)
        epoch_accs.append(avg_acc)
        print(f"平均损失: {avg_loss:.4f} | 平均准确率: {avg_acc:.4f}\n")

    return model, epoch_losses, epoch_accs

def classify_text_lstm(text, model, char_to_index, max_len, index_to_label):
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)

    _, predicted_index = torch.max(output, 1)
    predicted_index = predicted_index.item()
    predicted_label = index_to_label[predicted_index]

    return predicted_label

# --- Training and Prediction ---
lstm_dataset = CharLSTMDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(lstm_dataset, batch_size=32, shuffle=True)

embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)

index_to_label = {i: label for label, i in label_to_index.items()}

# 存储三个模型的训练结果
results = {}
models = {}

for model_type in range(1,4,1):
    model, loss, acc = train_and_evaluate(model_type, dataloader, vocab_size, embedding_dim, hidden_dim, output_dim, num_epochs=4)
    results[model_type] = {'loss': loss, 'acc': acc}
    models[model_type] = model
print("\n==================== 模型实验结果对比 ====================")
for model_type in range(1,4):  # 修正循环
    final_acc = results[model_type]['acc'][-1]
    final_loss = results[model_type]['loss'][-1]
    # 直接在打印里判断，输出对应模型名
    print(f"{'LSTM' if model_type==1 else 'RNN' if model_type==2 else 'GRU'} | 最终准确率: {final_acc:.4f} | 最终损失: {final_loss:.4f}")
print("\n==================== 模型预测测试 ====================")
model_type = 1
new_text = "帮我导航到北京"
predicted_class = classify_text_lstm(new_text, models[model_type], char_to_index, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text_lstm(new_text_2, models[model_type], char_to_index, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")
