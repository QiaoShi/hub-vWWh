import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader

# 1、数据加载
dataset = pd.read_csv("dataset.csv", sep='\t', header=None)
texts = dataset[0].tolist()
label_texts = dataset[1].tolist()
print(len(texts), len(label_texts))

label_to_index = {label: i for i, label in enumerate(set(label_texts))}
numerical_labels = [label_to_index[label] for label in label_texts]
char_to_index = {'<pad>': 0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)
index_to_char = {i: char for char, i in char_to_index.items()}
vocab_size = len(char_to_index)
print(vocab_size)

max_len = 30


class CharRNNDataset(Dataset):
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


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim):
        super(RNNClassifier, self).__init__()
        self.pooling = "mean"
        self.bidirectional = False
        self.num_directions = 1
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        embed = self.embedding(x)
        output, hn = self.rnn(embed)
        feat = torch.mean(output, dim=1)
        out = self.fc(feat)
        return out


lstm_dataset = CharRNNDataset(texts, numerical_labels, char_to_index, max_len)
dataloader = DataLoader(lstm_dataset, batch_size=32, shuffle=True)

embedding_dim = 128
hidden_dim = 128
output_dim = len(label_to_index)

model = RNNClassifier(vocab_size, embedding_dim, hidden_dim, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

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

    print(f"Epoch [{epoch + 1}/{num_epochs}], RNN Loss: {running_loss / len(dataloader):.4f}")

def classify_text_rnn(text, model, char_to_index, max_len, index_to_label):
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
predicted_class = classify_text_rnn(new_text, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text}' 预测为: '{predicted_class}'")

new_text_2 = "查询明天北京的天气"
predicted_class_2 = classify_text_rnn(new_text_2, model, char_to_index, max_len, index_to_label)
print(f"输入 '{new_text_2}' 预测为: '{predicted_class_2}'")
