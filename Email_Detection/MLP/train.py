import json
import numpy as np
from nltk_func import tokenize, stem, bag_of_words
from mlp_model import NeuralNet

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datetime import date
from datetime import datetime

today = date.today()
current_time = datetime.now()
current_time = current_time.strftime("%H_%M_%S")

with open('/Users/willis/Desktop/github repos/NYP_ITI110/Email_Detection/sample_dict.json', 'r') as f:
    intents = json.load(f)

all_words = []
tags = []
word_tag = []

for intent in intents['intents']:
    tag = intent['tag']
    tags.append(tag)
    for pattern in intent['patterns']:
        w = tokenize(pattern)
        all_words.extend(w)
        word_tag.append((w, tag))

ignore_words = ['?', '!', '.', ',']
all_words = [stem(w) for w in all_words if w not in ignore_words]
all_words = sorted(set(all_words))
tags = sorted(set(tags))

x_train = []
y_train = []

for (pattern, tag) in word_tag:
    bag = bag_of_words(pattern, all_words)
    x_train.append(bag)

    label = tags.index(tag)
    y_train.append(label)

x_train = np.array(x_train)
y_train = np.array(y_train)

class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(x_train)
        self.x_data = x_train
        self.y_data = y_train

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]
    
    def __len__(self):
        return self.n_samples

# hyperparameters
batch_size = 512
hidden_size = 1024
output_size = len(tags)
input_size = len(x_train[0])
learning_rate = 0.01
max_epochs = 1000

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f'Device: {device}')
model = NeuralNet(input_size, hidden_size, output_size).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(max_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.type(torch.LongTensor)
        labels = labels.to(device)

        outputs = model(words)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch+1) % 100 == 0:
        print(f"Epoch {epoch+1}/{max_epochs}, loss={loss.item():.4f}")

print(f"Final loss: loss={loss.item():.4f}")

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

FILE = "/Users/willis/Desktop/github repos/NYP_ITI110/Email_Detection/Data/data_{}_{}.pth".format(today, current_time)
torch.save(data, FILE)

print(f"Training complete, file saved to {FILE}")