import random
import json
import torch
import argparse
from lstm_model import LSTMClassifier
from nltk_func import bag_of_words, tokenize

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(device)

parser = argparse.ArgumentParser(description="Chatbot Model Loader")
parser.add_argument("model_path", type=str, help="Path to the model file (.pth)")
args = parser.parse_args()

with open('/Users/willis/Desktop/github repos/NYP_ITI110/Email_Detection/sample_dict.json', 'r') as f:
    intents = json.load(f)

# FILE = '/Users/willis/Desktop/github repos/NYP_ITI110/Email_Detection/Data/lstm_data_2025-02-26_08_49_14.pth'
FILE = args.model_path
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = LSTMClassifier(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sarah"
print("Let's chat! type 'quit' to exit")
while True:
    sentence = input('You: ')
    if sentence == 'quit':
        break

    sentence = tokenize(sentence)
    x = bag_of_words(sentence, all_words)
    x = x.reshape(1, x.shape[0])
    x = torch.from_numpy(x)

    output = model(x.to(device))

    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f'{bot_name}: I do not understand')