import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

dataset = pd.read_csv("../Week01/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

#标签转数字
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

#构建字符表串
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


class RNNFamilyClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim,rnn_type="lstm"):
        super().__init__()

        # 词表大小 转换后维度的维度
        self.embedding = nn.Embedding(vocab_size, embedding_dim) # 随机编码的过程， 可训练的
        self.rnn_type = rnn_type

        if self.rnn_type == "lstm":
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        elif self.rnn_type == "gru":
            self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        elif self.rnn_type == "rnn":
            self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        else:
            raise ValueError("rnn_type must be one of: 'rnn', 'lstm', 'gru'")# 循环层

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # batch size * seq length -》 batch size * seq length * embedding_dim
        embedded = self.embedding(x)

        '''# batch size * seq length * embedding_dim -》 batch size * seq length * hidden_dim
        gru_out, hidden_state = self.gru(embedded)

        # batch size * output_dim
        out = self.fc(hidden_state.squeeze(0))
        return out'''

        if self.rnn_type == "lstm":
            out_seq, (h, c) = self.rnn(embedded)
            last_h = h.squeeze(0)  # [B, H]
        else:
            out_seq, h = self.rnn(embedded)
            last_h = h.squeeze(0)  # [B, H]

        logits = self.fc(last_h)  # [B, C]
        return logits


# --- Training and Prediction ---
lstm_dataset = CharLSTMDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(lstm_dataset, batch_size=32, shuffle=True)

embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)
#######
results = {}
models = {}

for model_type in ["rnn", "lstm", "gru"]:
    print("\n" + "="*50)
    print(f"Now training: {model_type.upper()}")

    model = RNNFamilyClassifier(vocab_size, embedding_dim, hidden_dim, output_dim,rnn_type=model_type)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 4
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # 计算训练集 accuracy（最小改动版：不划分测试集）
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            if idx % 50 == 0:
                print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

        acc = correct / max(total, 1)
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")
    results[model_type] = acc
    models[model_type] = model

print("\n" + "="*50)
print("Accuracy Compare (train acc):")
for k in ["rnn", "lstm", "gru"]:
    print(f"{k.upper():4s}: {results[k]:.4f}")

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






作业二

四个模型比较：
一．Regex_rule.py（正则模型）

1.模型原理：
基于人工编写的正则表达式或者规则，通过字符串匹配直接判断文本类别，不涉及机器学习训练过程。

2.优点：
（1）实现简单：不需要训练，不需要数据集
（2）可解释性强：每一条规则都能解释清楚“为什么命中”
（3）运行速度快：纯字符串匹配，几乎无计算开销
（4）在强规则场景下非常稳定（如关键词过滤）

3.缺点：
（1）泛化能力极差：只能覆盖写过的规则
（2）维护成本高：规则一多就容易冲突，难管理
（3）无法处理语言多样性（同义、歧义、变体）
（4）对复杂语义完全无能为力

二．TF-IDF_ml.py（tfidf+传统机器学习）

1.模型原理：
还用TF-IDF将文本表示为稀缺向量，再用传统机器学习模型进行分类。

2.优点：
（1）实现成熟稳定
（2）训练速度快：对算力要求低
（3）对小数据集友好
（4）可解释性好（可以看哪些词权重高）

3.缺点：
（1）忽略词序和上下文
（2）无法建模语义关系
（3）对同义词、长距离依赖无能为力
（4）特征维度高，稀疏性强

三．Bert（预训练语言模型）

1.模型原理：
基于transformer的预训练语言模型，通过大模型语料学习上下文语义表示，再对下游文本分类任务进行微调。

2.优点：
（1）强大的语义理解能力
（2）能处理上下文，歧义，多义词
（3）在大多数NLP任务上效果最优
（4）迁移学习能力强

3.缺点：
（1）计算资源消耗大（显存，时间）
（2）模型结构复杂，不易调试
（3）推理速度慢
（4）可解释性弱

四．Prompt（prompt+大模型llm）

1.模型原理：
通过设计prompt，直接调用大预言模型进行灵验本或少样本分类，不进行显示训练

2.优点：
（1）几乎不需要训练数据
（2）上线速度极快
（3）对复杂语义理解能力强
（4）Prompt可快速迭代优化

3.缺点：
（1）如果不稳定（受prompt影响大）
（2）成本高（api调用）
（3）难以完全复现结果
（4）不适合大规模批量处理
