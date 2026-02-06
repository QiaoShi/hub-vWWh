import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset,DataLoader
import matplotlib.pyplot as plt

data = pd.read_csv('dataset.csv',sep='\t',header=None)
texts = data[0].tolist()
labels = data[1].tolist()

label_to_index = {label: i for i, label in enumerate(set(labels))}
numerical_labels = [label_to_index[label] for label in labels]
num_label = len(label_to_index)


char_to_index = {'pad':0}
for text in texts:
    for char in text:
        if char not in char_to_index:
            char_to_index[char] = len(char_to_index)
vocab_size = len(char_to_index)
max_len = 40


class textClassifyDataset(Dataset):
    def __init__(self,texts, numerical_label, char_to_index, max_len):
        self.texts = texts
        self.char_to_index = char_to_index
        self.label = torch.tensor(numerical_label,dtype = torch.long)
        self.max_len = max_len 

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        input_texts = self.texts[index]
        indices = [self.char_to_index.get(char,0) for char in input_texts[:self.max_len]]
        indices += [0]*(self.max_len-len(indices))
    
        return torch.tensor(indices,dtype=torch.long), self.label[index] #batch_size*max_len

dataset = textClassifyDataset(texts, numerical_labels, char_to_index, max_len)
batch_size = 32
dataloader = DataLoader(dataset, batch_size=batch_size,shuffle=True)
# RNN模型
class rnn_Module(nn.Module):
    def __init__(self,input_dim, embedding_dim, hidden_dim, output_dim):
        super(rnn_Module, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        # batch_size*max_len -> batch_size*max_len*embedding_dim
        self.embedding = nn.Embedding(self.input_dim, self.embedding_dim)
        #batch_size*max_len*embedding_dim -> batch_size*max_len*hidden_dim
        self.rnn = nn.RNN(self.embedding_dim,self.hidden_dim, batch_first=True)
        # batch_size*max_len*hidden_dim -> batch_size*max_len*output_size
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, x):
        embedd_out = self.embedding(x)
        # hn= max_len*batch_size*hidden_dim   lstm_out = batch_size*max_len*hidden_dim  
        lstm_out, hn = self.rnn(embedd_out) 
        # out = self.linear(hn.squeeze(0))
        out = self.linear(hn.squeeze(0))
        return out
# LSTM模型
class lstm_Module(nn.Module):
    def __init__(self,input_dim, embedding_dim, hidden_dim, output_dim):
        super(lstm_Module, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        # batch_size*max_len -> batch_size*max_len*embedding_dim
        self.embedding = nn.Embedding(self.input_dim, self.embedding_dim)
        #batch_size*max_len*embedding_dim -> batch_size*max_len*hidden_dim
        self.lstm = nn.LSTM(self.embedding_dim,self.hidden_dim, batch_first=True)
        # batch_size*max_len*hidden_dim -> batch_size*max_len*output_size
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, x):
        embedd_out = self.embedding(x)
        # hn= max_len*batch_size*hidden_dim   lstm_out = batch_size*max_len*hidden_dim  
        lstm_out, (hn,cn) = self.lstm(embedd_out) 
        # out = self.linear(hn.squeeze(0))
        out = self.linear(hn.squeeze(0))
        return out
# GRU 模型 
class gru_Module(nn.Module):
    def __init__(self,input_dim, embedding_dim, hidden_dim, output_dim):
        super(gru_Module, self).__init__()
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        # batch_size*max_len -> batch_size*max_len*embedding_dim
        self.embedding = nn.Embedding(self.input_dim, self.embedding_dim)
        #batch_size*max_len*embedding_dim -> batch_size*max_len*hidden_dim
        self.gru = nn.GRU(self.embedding_dim,self.hidden_dim, batch_first=True)
        # batch_size*max_len*hidden_dim -> batch_size*max_len*output_size
        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, x):
        embedd_out = self.embedding(x)
        # hn= max_len*batch_size*hidden_dim   lstm_out = batch_size*max_len*hidden_dim  
        lstm_out, hn = self.gru(embedd_out) 
        # out = self.linear(hn.squeeze(0))
        out = self.linear(hn.squeeze(0))
        return out


embedding_dim = 64
hidden_dim = 128
output_dim = len(label_to_index)
epochs = 10
loss_model_type = {}
def train_model(model_type):
    if model_type == 'rnn':
        model = rnn_Module(vocab_size, embedding_dim, hidden_dim, output_dim)     
    if model_type == 'lstm':
        model = lstm_Module(vocab_size, embedding_dim, hidden_dim, output_dim)
    if model_type == 'gru':
        model = gru_Module(vocab_size, embedding_dim, hidden_dim, output_dim)
    fn_loss = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr = 0.001)
    loss_epoch = []
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for i, (inputs, labels) in enumerate(dataloader):
            optimizer.zero_grad()
            y = model(inputs)
            loss = fn_loss(y, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if i % 50 == 0:
                print(f"Batch 个数 {i}, 当前Batch Loss: {loss.item()}")

        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(dataloader):.4f}")
        loss_epoch.append(round(running_loss / len(dataloader),4))
    loss_model_type[model_type] = loss_epoch

plt.figure(figsize=(10,6))
model_type = ['rnn', 'lstm', 'gru']
for mtype in model_type:
    train_model(mtype)
    plt.plot(loss_model_type[mtype], label=mtype)
plt.xlabel('epoch num')
plt.ylabel('loss')
plt.title('Accuracy eval')
plt.legend()
plt.show()


