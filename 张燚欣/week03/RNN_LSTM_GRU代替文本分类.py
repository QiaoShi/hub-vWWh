import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import time
from sklearn.metrics import accuracy_score

# 读取数据
dataset = pd.read_csv("D:/badou/第3周：序列模型与语言模型/Week03/作业/dataset.csv", sep="\t", header=None)
texts = dataset[0].tolist()
string_labels = dataset[1].tolist()

# 标签编码
label_to_index = {label: i for i, label in enumerate(set(string_labels))}
numerical_labels = [label_to_index[label] for label in string_labels]

# 字符编码
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)

index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)

max_len = 40


# 数据集类
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


# 创建数据集
full_dataset = CharRNNDataset(texts, numerical_labels, char_to_index, max_len)

# 划分训练集和测试集（8:2）
train_size = int(0.8 * len(full_dataset))
test_size = len(full_dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# --- RNN分类器 ---
class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        rnn_out, hidden_state = self.rnn(embedded)
        out = self.fc(hidden_state.squeeze(0))
        return out


# --- LSTM分类器 ---
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(LSTMClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, (hidden_state, cell_state) = self.lstm(embedded)
        out = self.fc(hidden_state.squeeze(0))
        return out


# --- GRU分类器 ---
class GRUClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(GRUClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embedded = self.embedding(x)
        gru_out, hidden_state = self.gru(embedded)
        out = self.fc(hidden_state.squeeze(0))
        return out


# 训练函数
def train_model(model, model_name, train_loader, test_loader, num_epochs=10):
    print(f"\n{'=' * 50}")
    print(f"开始训练 {model_name}")
    print(f"{'=' * 50}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []
    train_accuracies = []
    test_accuracies = []

    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        all_preds = []
        all_labels = []

        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # 计算准确率
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        train_loss = running_loss / len(train_loader)
        train_acc = accuracy_score(all_labels, all_preds)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        # 测试阶段
        model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in test_loader:
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        test_acc = accuracy_score(all_labels, all_preds)
        test_accuracies.append(test_acc)

        print(f"Epoch [{epoch + 1}/{num_epochs}] - "
              f"Train Loss: {train_loss:.4f}, "
              f"Train Acc: {train_acc:.4f}, "
              f"Test Acc: {test_acc:.4f}")

    # 最终测试准确率
    final_test_acc = test_accuracies[-1]

    return train_losses, train_accuracies, test_accuracies, final_test_acc


# 预测函数
def classify_text(text, model, char_to_index, max_len, index_to_label):
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


# 超参数设置
embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)
num_epochs = 10

index_to_label = {i: label for label, i in label_to_index.items()}

# 分别训练三个模型
results = {}

# 训练RNN
rnn_model = RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
rnn_results = train_model(rnn_model, "RNN", train_loader, test_loader, num_epochs)
results['RNN'] = rnn_results

# 训练LSTM
lstm_model = LSTMClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
lstm_results = train_model(lstm_model, "LSTM", train_loader, test_loader, num_epochs)
results['LSTM'] = lstm_results

# 训练GRU
gru_model = GRUClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
gru_results = train_model(gru_model, "GRU", train_loader, test_loader, num_epochs)
results['GRU'] = gru_results

# 打印对比结果
print(f"\n{'=' * 60}")
print("模型性能对比")
print(f"{'=' * 60}")
print(f"{'模型':<10} {'最终测试准确率':<20}")
print(f"{'-' * 30}")
for model_name, (train_losses, train_accs, test_accs, final_test_acc) in results.items():
    print(f"{model_name:<10} {final_test_acc:.4f}")
print(f"{'=' * 60}")

# 测试新样本
test_samples = [
    "帮我导航到江苏",
    "杭州的天气怎么样",
    "播放周杰伦的音乐",
    "打开客厅的灯",
    "导航去高铁站",
    "明天会下雨吗"
]

print(f"\n{'=' * 60}")
print("新样本预测结果")
print(f"{'=' * 60}")

for sample in test_samples:
    print(f"\n输入文本: '{sample}'")
    print("-" * 40)
    for model_name, model_info in [('RNN', rnn_model), ('LSTM', lstm_model), ('GRU', gru_model)]:
        model = model_info
        prediction = classify_text(sample, model, char_to_index, max_len, index_to_label)
        print(f"{model_name}预测: {prediction}")