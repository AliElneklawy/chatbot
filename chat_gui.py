# ==================== Imports ====================

import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
import json
from string import punctuation
import pickle
import random
from tensorflow.keras.models import load_model

# ==================== Loading Data =====================

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
intents = json.loads(open('intents.json').read())
model = load_model('chatbot_model.h5')

# ==================== Functions =========================

lemmatizer = WordNetLemmatizer()
def cleanup_sentence(sentence: str) -> list:
    tokenized = nltk.word_tokenize(sentence)
    lemmatized = [lemmatizer.lemmatize(word.lower()) for word in tokenized]
    return lemmatized

def bag_of_words(words: list, sentence_words: list) -> np.array:
    bag = [0] * len(words)
    for sentence_word in sentence_words:
        for i, word in enumerate(words):
            if sentence_word == word:
                bag[i] = 1
    return np.array(bag)

def predict_class(msg: np.array) -> list:
    msg_class_probs = model.predict(np.array([msg]), verbose=0)[0]
    results = [[i,r] for i, r in enumerate(msg_class_probs)]
    results.sort(key=lambda x: x[1], reverse=True)
    return results

def get_response(msg_class: str) -> str:
    for tag in intents['intents']:
        if tag['tag'] == msg_class:
            random_response = random.choice(tag['responses'])
    return random_response

def google_response():
    import webbrowser

    url = "https://www.google.com"
    webbrowser.open(url)


def footballnews_response():
    pass

def footballmatches_response():
    pass

def news_response():
    pass

def datetime_response() -> str:
    import datetime

    current_datetime = datetime.datetime.now()
    formatted_datetime = current_datetime.strftime("\n  Date: %Y-%m-%d\n  Time: %H:%M:%S")
    return formatted_datetime


def main(msg: str) -> tuple[str, str]:
    msg = cleanup_sentence(msg)
    msg = bag_of_words(words, msg)
    msg_class_probs = predict_class(msg)
    msg_class = classes[msg_class_probs[0][0]]
    response = get_response(msg_class)
    return response, msg_class

# ==================== Demo =====================

while True:
    msg = input(">>> ")
    response, msg_class = main(msg)

    if msg_class == 'google':
        print('Bot: Redirecting to google...')
        google_response()

    elif msg_class == 'footballnews':
        footballnews_response()
    elif msg_class == 'footballmatches':
        footballmatches_response()
    elif msg_class == 'news':
        news_response()
    elif msg_class == 'datetime':
        current_datetime = datetime_response()
        print("Bot:", current_datetime)

    elif msg_class == 'goodbye':
        print("Bot: " + response)
        break

    else:
        print("Bot: " + response)
