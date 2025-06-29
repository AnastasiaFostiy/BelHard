import random
import json
import torch
from model import NeuralNet
from nltk_utils import bag_of_words, tokenize


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

sentences = []
bot_name = "IRecommend"
sentences.append(f"{bot_name}: Let's talk! (type 'The end' to exit)")
print(sentences[-1])

while True:
    sentence = input("You: ")
    sentences.append(f"You: {sentence}")
    if sentence.lower() == "the end":
        sentences.append(f"{bot_name}: <3...")
        print(sentences[-1])
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent["tag"]:
                sentences.append(f"{bot_name}: {random.choice(intent['responses'])}")
                print(sentences[-1])
    else:
        sentences.append(f"{bot_name}: Pu-pu-pu, lets change the subject..!")
        print(sentences[-1])


with open('IRecommendBot_history.txt', 'w') as file:
    for line in sentences:
        file.write(line + '\n')

