import torch
import torch.nn as nn
import pandas as pd
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

dataset = pd.read_csv('../dataset.csv', sep='\t', header=None, nrows=None)
text_list = dataset[0].tolist()
labels = dataset[1].tolist()

label_to_idx = {label: idx for idx, label in enumerate(set(labels))}
numerical_idx = [label_to_idx[label] for label in labels]
idx_to_label = {idx: label for label, idx in label_to_idx.items()}

char_to_idx = {'<pad>': 0}
for text in text_list:
    for char in text:
        if char not in char_to_idx:
            char_to_idx[char] = len(char_to_idx)
idx_to_char = {idx: char for char, idx in char_to_idx.items()}

vocab_size = len(char_to_idx)
max_len = 40

class CharBowDataset(Dataset):
    def __init__(self, texts, labels, max_len, char_to_idx, vocab_size):
        self.texts = texts
        self.labels = torch.tensor(labels, dtype=torch.long)
        self.char_to_idx = char_to_idx
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.bow_vectors = self._create_bow_vectors()

    def _create_bow_vectors(self):
        tokenizers = []
        for text in self.texts:
            tokenizer = [self.char_to_idx.get(char, 0) for char in text[:max_len]]
            tokenizer += [0] * (max_len - len(tokenizer))
            tokenizers.append(tokenizer)

        bow_vectors = []
        for tokenizer in tokenizers:
            bow_vector = torch.zeros(self.vocab_size)
            for index in tokenizer:
                if index != 0:
                    bow_vector[index] += 1
            bow_vectors.append(bow_vector)
        return torch.stack(bow_vectors)


    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        return self.bow_vectors[idx], self.labels[idx]

char_dataset = CharBowDataset(text_list, numerical_idx, max_len, char_to_idx, vocab_size)
dataloader = DataLoader(char_dataset, batch_size=32, shuffle=True)

class ClassifyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim1, output_dim):
        super(ClassifyModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, output_dim),
        )

    def forward(self, x):
        return self.network(x)

input_dim = vocab_size
hidden_dim1, output_dim = 32, len(label_to_idx)
model = ClassifyModel(input_dim, hidden_dim1, output_dim)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

num_epochs = 10
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
            with open('一层隐藏层，每层有32个神经元的loss变化.txt', 'a', encoding='utf-8') as f:
                f.write(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}\n")
            print(f"Batch 个数 {idx}, 当前Batch Loss: {loss.item()}")

    with open('一层隐藏层，每层有32个神经元的loss变化.txt', 'a', encoding='utf-8', newline='\n') as f:
        f.write(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}]\n")
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}]")

def classify_text(new_text, model):
    model.eval()
    tokenizer = [char_to_idx.get(char, 0) for char in new_text[:max_len]]
    tokenizer += [0] * (max_len - len(tokenizer))
    bow_vector = torch.zeros(vocab_size)
    for index in tokenizer:
        if index != 0:
            bow_vector[index] += 1
    bow_vector = bow_vector.unsqueeze(0)

    model.eval()
    with torch.no_grad():
        outputs = model(bow_vector)

    _, predict_index = torch.max(outputs, 1)
    predict_index = predict_index.item()
    predict_label = idx_to_label[predict_index]

    return predict_label

new_text = '帮我导航去山东'
predict_class = classify_text(new_text, model)
print(f'输入文本{new_text}, 分类为{predict_class}')

new_text = '查询明天山东的天气'
predict_class = classify_text(new_text, model)
print(f'输入文本{new_text}, 分类为{predict_class}')
