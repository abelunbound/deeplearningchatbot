import random
import json
import pickle
import numpy as np

import nltk
from nltk.stem import WordNetLemmatizer

from tensorflow.keras.models import load_model

lemmatizer = WordNetLemmatizer()
#I got an error here,TypeError(f'the JSON object must be str, bytes or bytearray,
# First solution was to use json.dump but it didn't work so the advise was to use
# json.dumps with an 's' as I learnt that json.dump requires a file to dump into
# while json.dumps is happy to dump into a variable
# intents_str = json.dumps(open('intents.json'))
intents1 = json.dumps(open('intents.json').read())
intents = json.loads(open('intents.json').read())

words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbotmodel.h5')

# Set up several functions to use the model the right way to convert numerical data to words
# 1. Function for cleaning up the sentences
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

# 2. Function for getting the bag of words
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    #Initial bag of zeros for as many words
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

# 3. Function for predicting the class based on the sentence
def predict_class(sentence):
    bow = bag_of_words(sentence)
    #res here is results
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]

    results.sort(key=lambda x:[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({'intent': classes[r[0]], 'probability': str(r[1])})
    return return_list

# 4. Function for getting the response
def get_response(intent_list, intents_json):
    tag = intent_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return  result

print('Yes! Our ChatBot, Mrs. Elizibeth is now running!')

while True:
    message = input('User:' + ' ')
    ints = predict_class(message)
    res = get_response(ints,intents)
    print('Lizzy:', res)