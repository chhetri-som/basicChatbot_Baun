import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy
import tensorflow
import tflearn
import random
import json
import pickle
import os

stemmer = LancasterStemmer()
# nltk.download('punkt')  # Only needed once

# Load intents file
with open("intents.json") as file:
    data = json.load(file)

# Load or preprocess data
try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except FileNotFoundError:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

    words = [stemmer.stem(w.lower()) for w in words if w not in "?"]
    words = sorted(list(set(words)))
    labels = sorted(labels)

    training = []
    output = []
    out_empty = [0 for _ in range(len(labels))]

    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            bag.append(1 if w in wrds else 0)

        output_row = out_empty[:]
        output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

# Model definition
net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)
model = tflearn.DNN(net)

# Check if model file exists
model_file = "model.tflearn"
meta_file = model_file + ".meta"

should_retrain = False

# Check if the model exists and ask if the user wants to retrain
if os.path.exists(meta_file):
    choice = input("Model already exists. Do you want to retrain the model? (yes/no): ").strip().lower()
    if choice == "no":
        try:
            model.load(model_file)
            print("✅ Model loaded successfully.")
            should_retrain = False
        except Exception as e:
            print("⚠️ Error loading model:", e)
            print("ℹ️ Proceeding to retrain the model due to loading error.")
            should_retrain = True
    else:
        print("ℹ️ Retraining the model as per user choice...")
        should_retrain = True
else:
    print("ℹ️ Model file not found. Training a new model...")
    should_retrain = True

if should_retrain:
    print("--- Starting Model Training ---")
    model.fit(training, output, n_epoch=500, batch_size=8, show_metric=True)
    model.save(model_file)
    print("Model trained and saved.")

# Chatbot functions
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]
    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
    return numpy.array(bag)

def chat():
    print("Start chatting with the bot (type 'quit' to exit):")
    last_tag = None
    last_response = []

    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])[0]
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        if results[results_index] > 0.7:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
            #Debug
            print(f"Confidence scores: {results}")
            print(f"Predicted intent: {tag} with confidence {results[results_index]:.2f}")
            print("Baun:", random.choice(responses))
        else:
            print("Baun: My apologies, could you maybe rephrase that?")

chat()
