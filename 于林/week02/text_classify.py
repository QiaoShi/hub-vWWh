import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from torch.utils.data import Dataset, DataLoader

# ... (Data loading and preprocessing remains the same) ...
dataset = pd.read_csv("dataset.csv", sep="\t", header=None)
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

max_len = 40
arch_config = [
    {'name': 'layer-1-unit-32', 'units': [32]},
    {'name': 'layer-1-unit-64', 'units': [64]},
    {'name': 'layer-2-unit-64-32', 'units': [64, 32]},
    {'name': 'layer-2-unit-128-64', 'units': [128, 64]},
    {'name': 'layer-3-unit-128-64-32', 'units': [128, 64, 32]},
    {'name': 'layer-3-unit-256-128-64', 'units': [256, 128, 64]},
    {'name': 'layer-4-unit-256-128-64-32', 'units': [256, 128, 64, 32]}
]

class CharBoWDataset(Dataset):
    def __init__(self, texts, labels, char_to_index, max_len, vocab_size):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_index = char_to_index
        self.max_len = max_len
        self.vocab_size = vocab_size
        self.bow_vectors = self._create_bow_vectors()

    def _create_bow_vectors(self):
        tokenized_texts = []
        for text in self.texts:
            tokenized = [self.char_to_index.get(char, 0) for char in text[:self.max_len]]
            tokenized += [0] * (self.max_len - len(tokenized))
            tokenized_texts.append(tokenized)

        bow_vectors = []
        for text_indices in tokenized_texts:
            bow_vector = torch.zeros(self.vocab_size)
            for index in text_indices:
                if index != 0:
                    bow_vector[index] += 1
            bow_vectors.append(bow_vector)
        return torch.stack(bow_vectors)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, hidden_units, out_dim):  # 层的个数 和 验证集精度
        # 层初始化
        super(SimpleClassifier, self).__init__()
        self.layers = nn.ModuleList()
        pre_units = input_dim
        for units in hidden_units:
            self.layers.append(nn.Linear(pre_units, units))
            self.layers.append(nn.ReLU())
            pre_units = units
        self.output = nn.Linear(units, out_dim)

    def forward(self, x):
        # 手动实现每层的计算
        for layer in self.layers:
            x = layer(x)
        return torch.sigmoid(self.output(x))


char_dataset = CharBoWDataset(texts, numerical_labels, char_to_index, max_len, vocab_size) # 读取单个样本
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True) # 读取批量数据集 -》 batch数据

arch_loss = {}

for arch in arch_config:
    output_dim = len(label_to_index)
    model = SimpleClassifier(vocab_size, arch['units'], output_dim)  # 维度和精度有什么关系？
    criterion = nn.CrossEntropyLoss() # 损失函数 内部自带激活函数，softmax
    optimizer = optim.SGD(model.parameters(), lr=0.01)

    # epoch： 将数据集整体迭代训练一次
    # batch： 数据集汇总为一批训练一次

    num_epochs = 10
    epoch_loss = []
    for epoch in range(num_epochs):  # 12000， batch size 100 -》 batch 个数： 12000 / 100
        model.train()
        running_loss = 0.0
        for idx, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            # if idx % 50 == 0:
            #     print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")
        epoch_loss.append((running_loss / len(dataloader)))
        # print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")
    arch_loss[arch['name']] = epoch_loss
    print(f"{arch['name']}_loss: {arch_loss[arch['name']]}")

plt.figure(figsize=(10, 6))
for arch in arch_config:
    plt_loss = arch_loss[arch['name']]
    plt.plot(plt_loss, label=arch['name'])
    plt.legend('layer-32')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.show()





#
#     plt_arch_classify_loss(arch, arch_loss)
#
# plt.a
# plt.show()
