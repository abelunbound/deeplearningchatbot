# Natural Language Processing and Neural Networks

# random module helps randomize the chatbot's response from a pool of responses.
import random
import json
import pickle
import numpy as np

import nltk

# I had two errors: first was a TypeError saying bool object is not iterable some
# suggestions on stackoverflow recommends converting the variable to string
# after converting with str() the code threw up a new error 'wordnet' not found
# it provided a command to download the 'wordnet' as seen below but a new error was thrown up
# a command was also provided to download it into the toolkit as shown below.
nltk.download('wordnet')
nltk.download('omw-1.4')

from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout
from tensorflow.keras.optimizers import SGD

# Call the constructor for Word lemmatizer from NLTK and store the constructor in a variable
lemmatizer = WordNetLemmatizer()
intents = json.loads(open('intents.json').read())

# These variables are created as global variables so they can be used else where?
# This variable will hold all the tokenized words in each of the users possible questions
words = []

# This variable holds/MAPs a list-pair containing a tupple which holds a combination of the user's intention and possible
# questions for that intention
documents = []
classes = []

ignore_letters = ['?', '!', '.', ',']

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# words = [WordNetLemmatizer().lemmatize(token_occurrence)
#          for token_occurrence in user_enquiry_pattern not in ignore_letters]

words = [lemmatizer.lemmatize(str(word)) for word in words if word not in ignore_letters]
# Eliminate duplicates and turn it back into a list
words = sorted(set(words))

# There may be no need to sort this variable below (which is equivalent to user intentiosn in 'tag'
classes = sorted(set(classes))

# What are pickle files?
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

# Machine Learning part

# Final step of preprocessing
# Get numbers to feed numbers into neural netwoork

# 1. Create a training list
training = []

# 2. Describe this function
output_empty = [0] * len(classes)

# 3. Write logic

for document in documents:
    bag = []
    word_patterns = document[0]
    word_patterns = [lemmatizer.lemmatize(word.lower()) for word in word_patterns]
    for word in words:
        bag.append(1) if word in word_patterns else bag.append(0)

    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# Final step of preprocessing

random.shuffle(training)
training = np.array(training)

train_x = list(training[:, 0])
train_y = list(training[:, 1])

#Start building a simple neural network.
#
# Create a simple sequential model
model = Sequential()
# Layer1: input layer
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
# Layer2: to prevent overfitting
model.add(Dropout(0.5))
# Layer3: Another dense layer
model.add(Dense(64, activation='relu'))
# Layer4: A dropout layer
model.add(Dropout(0.5))
# Layer5: Dense layer for as many as there are labels?
model.add(Dense(len(train_y[0]), activation='softmax'))

#Stochastic Gradient Descent, SGD (an optmizer)

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

#Compile the model
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

#Fit the model. Epoch is the number of times we want to feed the data into the model
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

#Save model
model.save('chatbotmodel.h5', hist)

print('Done with Training')

