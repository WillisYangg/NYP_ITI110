import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from datetime import date, datetime
from nltk_func import tokenize, stem, bag_of_words
from lstm_model import LSTMClassifier

today = date.today()
current_time = datetime.now().strftime("%H_%M_%S")

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
    y_train.append(tags.index(tag))

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

# Hyperparameters
batch_size = 64
hidden_size = 128
num_layers = 2
output_size = len(tags)
input_size = len(x_train[0])
learning_rate = 0.001
max_epochs = 1000

dataset = ChatDataset()
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=0)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Device: {device}")
model = LSTMClassifier(input_size, hidden_size, output_size, num_layers).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(max_epochs):
    for (words, labels) in train_loader:
        words = words.to(device)
        labels = labels.to(device)

        outputs = model(words.unsqueeze(1).float())
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1}/{max_epochs}, Loss={loss.item():.4f}")

print(f"Final Loss: {loss.item():.4f}")

# Save model
data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "output_size": output_size,
    "hidden_size": hidden_size,
    "all_words": all_words,
    "tags": tags
}

FILE = f"/Users/willis/Desktop/github repos/NYP_ITI110/Email_Detection/Data/lstm_data_{today}_{current_time}.pth"
torch.save(data, FILE)

print(f"Training complete, model saved to {FILE}")