import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- 1. 数据准备 ---

dataset = pd.read_csv("../../../../lesson1/Week01/Week01/dataset.csv", sep="\t", header=None)

texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

# 标签转索引
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
index_to_label = {i: label for label, i in label_to_index.items()}
numerical_labels = [label_to_index[label] for label in string_labels]

# 字符转索引 (Char-level)
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

vocab_size = len(char_to_index)
max_len = 40

# --- 2. Dataset 类 ---
class CharRNNDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long), self.labels[idx]

# --- 3. 统一的模型类 (核心修改部分) ---
class GenericRNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, rnn_type='LSTM'):
        super(GenericRNNClassifier, self).__init__()
        self.rnn_type = rnn_type.upper()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        
        # 理由：三者初始化接口一致，通过判断实现无缝切换
        if self.rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        elif self.rnn_type == 'GRU':
            self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)

        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        # 理由：LSTM 返回 (out, (h, c))，RNN/GRU 返回 (out, h)
        output, hn = self.rnn(embedded)
        
        # 取最后一个隐藏层状态
        # 如果是 LSTM，hn 是 (h_n, c_n) 元组，取第一个
        last_hidden = hn[0] if self.rnn_type == 'LSTM' else hn
        
        # last_hidden 形状: (num_layers * num_directions, batch, hidden_dim)
        # 这里 num_layers=1, 所以 squeeze(0) 得到 (batch, hidden_dim)
        out = self.fc(last_hidden.squeeze(0))
        return out

# --- 4. 训练与对比实验 ---
def train_and_evaluate(model_type, train_loader, params):
    model = GenericRNNClassifier(
        params['vocab_size'], 
        params['embedding_dim'], 
        params['hidden_dim'], 
        params['output_dim'], 
        rnn_type=model_type
    )
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(params['epochs']):
        model.train()
        total_loss = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
    
    # 简单的精度评估 (这里直接用训练集模拟评估逻辑)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in train_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return accuracy, model

# --- 5. 执行实验 ---
configs = {
    'vocab_size': vocab_size,
    'embedding_dim': 64,
    'hidden_dim': 128,
    'output_dim': len(label_to_index),
    'epochs': 10
}

train_dataset = CharRNNDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)

results = {}
models = {}

for m_type in ['RNN', 'LSTM', 'GRU']:
    acc, trained_model = train_and_evaluate(m_type, dataloader, configs)
    results[m_type] = acc
    models[m_type] = trained_model
    print(f"模型 {m_type:4} 训练完成，准确率: {acc:.4f}")

# --- 6. 预测函数 ---
def predict(text, model, char_to_index, max_len, index_to_label):
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)
    return index_to_label[pred.item()]

# 测试预测
test_text = "帮我导航到北京"
print(f"\n测试输入: '{test_text}'")
for m_type, m_obj in models.items():
    res = predict(test_text, m_obj, char_to_index, max_len, index_to_label)
    print(f"[{m_type}] 预测结果: {res}")
