# 理解rnn、lstm、gru的计算过程（面试用途），阅读官方文档 ：https://docs.pytorch.org/docs/2.4/nn.html#recurrent-layers 最终 使用 GRU 代替 LSTM 实现05_LSTM文本分类.py；05_LSTM文本分类.py 使用lstm ，使用rnn/ lstm / gru 分别代替原始lstm，进行实验，对比精度


'''
rnn:
RNN 就是在时间维度上“重复用同一组参数”的全连接网络
每读入一个 token就把 当前输入和 上一个时刻的隐藏状态
拼在一起算一个新的隐藏状态
容易梯度爆炸/消失 ，其根本原因在于反向传播过程中梯度的连乘结构，结果：早期时间步的梯度趋近于 0 → 参数几乎不更新 → 网络无法学习长期依赖。
所以一开始我很难调整到正确的，然后为了添加上下文改成双向的就可以用正确的结果

LSTM/GRU 解决梯度消失，让模型能学长期依赖
LSTM 的核心思想是引入一个细胞状态（cell state） 
它像一条“高速公路”，信息可以在其中几乎无损地流动。通过三个“门”来控制这条高速路的信息更新与输出：
遗忘门（Forget Gate）：决定丢弃哪些旧信息
输入门（Input Gate）：决定添加哪些新信息
输出门（Output Gate）：决定输出什么

GRU 将细胞状态和隐藏状态合并，只有两个门：
重置门（Reset Gate）：控制如何忽略过去信息
更新门（Update Gate）：控制新旧信息的混合比例
gru比lstm收敛更快些
'''



import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

dataset = pd.read_csv("../../dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

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
        # pad and crop
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]

# a = CharLSTMDataset()
# len(a) -> a.__len__
# a[0] -> a.__getitem__


# --- NEW LSTM Model Class ---


class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()

        # 词表大小 转换后维度的维度
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # 随机编码的过程， 可训练的
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)  # 循环层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # batch size * seq length -》 batch size * seq length * embedding_dim
        embedded = self.embedding(x)

        # batch size * seq length * embedding_dim -》 batch size * seq length * hidden_dim
        lstm_out, (hidden_state, cell_state) = self.lstm(embedded)

        # batch size * output_dim
        out = self.fc(hidden_state.squeeze(0))
        return out


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNClassifier, self).__init__()

        # 词表大小 转换后维度的维度
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # 随机编码的过程， 可训练的
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True,nonlinearity='relu',bidirectional=True)  # 循环层
        # self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True,nonlinearity='relu')  # 它在处理长序列数据时会遇到 梯度消失 或 梯度爆炸 的问题
        self.fc = nn.Linear(hidden_dim*2, output_dim)
        self.dropout = nn.Dropout(0.3)

        
    def forward(self, x):
        # batch size * seq length -》 batch size * seq length * embedding_dim
        embedded = self.embedding(x)

        # batch size * seq length * embedding_dim -》 batch size * seq length * hidden_dim
        rnn_out, hn = self.rnn(embedded)
        
        # batch size * output_dim
        hn = hn[-2:].transpose(0, 1).contiguous().view(hn.shape[1], -1)

        # 添加Dropout
        out = self.dropout(hn.squeeze(0))

        # batch size * output_dim
        out = self.fc(hn)
        return out


class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GRUClassifier, self).__init__()

        # 词表大小 转换后维度的维度
        self.embedding = nn.Embedding(vocab_size, embedding_dim)  # 随机编码的过程，可训练的
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True,)  # 使用GRU代替RNN或LSTM
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # batch size * seq length -》 batch size * seq length * embedding_dim
        embedded = self.embedding(x)

        # batch size * seq length * embedding_dim -》 batch size * seq length * hidden_dim
        gru_out, hidden_state = self.gru(embedded)

        # batch size * output_dim
        out = self.fc(hidden_state.squeeze(0))  # 这里使用最后的隐藏状态进行分类
        return out




# --- Training and Prediction ---
lstm_dataset = CharLSTMDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(lstm_dataset, batch_size=64, shuffle=True)

embedding_dim = 64
hidden_dim = 256
output_dim = len(label_to_index)

# model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
model = RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
# model = GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-5)

num_epochs = 4
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for idx, (inputs, labels) in enumerate(dataloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if idx % 50 == 0:
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

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

index_to_label = {i: label for label, i in label_to_index.items()}

new_text = "帮我导航到北京"
predicted_class = classify_text_lstm(new_text, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text_lstm(new_text_2, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")