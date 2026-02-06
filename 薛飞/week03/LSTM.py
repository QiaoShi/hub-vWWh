import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# 设备自动适配
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"使用设备：{device}")

# 1. 数据加载与预处理（完全复用）
dataset = pd.read_csv("../Week03/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]
index_to_label = {i: label for label, i in label_to_index.items()}
output_dim = len(label_to_index)

char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)
vocab_size = len(char_to_index)
max_len = 40

# 2. 自定义Dataset（适配设备）
class CharLSTMDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long).to(device)
        self.char_to_index = char_to_index
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        indices = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
        indices += [0] * (self.max_len - len(indices))
        return torch.tensor(indices, dtype=torch.long).to(device), self.labels[idx]

# 3. 定义标准LSTM模型（你原始的核心逻辑，仅加设备适配）
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)  # 原始LSTM层
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden_state, cell_state) = self.lstm(embedded)  # LSTM返回(输出, (隐状态, 细胞状态))
        out = self.fc(hidden_state.squeeze(0))  # 取隐状态做分类
        return out

# 4. 数据加载器+模型实例化
lstm_dataset = CharLSTMDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(lstm_dataset, batch_size=32, shuffle=True)

embedding_dim = 64
hidden_dim = 128
model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim).to(device)  # 移到设备
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. 训练循环（完全复用）
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
            print(f"Batch {idx}, Loss: {loss.item():.4f}")
    print(f"Epoch [{epoch+1}/{num_epochs}], Avg Loss: {running_loss/len(dataloader):.4f}\n")

# 6. 预测函数（适配设备）
def classify_text_lstm(text, model, char_to_index, max_len, index_to_label, device):
    indices = [char_to_index.get(char, 0) for char in text[:max_len]]
    indices += [0] * (max_len - len(indices))
    input_tensor = torch.tensor(indices, dtype=torch.long).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
    _, predicted_idx = torch.max(output, 1)
    return index_to_label[predicted_idx.item()]

# 测试预测
new_text1 = "帮我导航到北京"
new_text2 = "查询明天北京的天气"
print(f"输入'{new_text1}' → 预测：{classify_text_lstm(new_text1, model, char_to_index, max_len, index_to_label, device)}")
print(f"输入'{new_text2}' → 预测：{classify_text_lstm(new_text2, model, char_to_index, max_len, index_to_label, device)}")
