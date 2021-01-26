# -*- coding: utf-8 -*-
"""
Created on Tue Jan 26 08:37:43 2021

@author: Aju
"""
#Text Preprocessing
#Use NLTK

#SPAM - 1 HAM - 0


import numpy as np
import pandas as pd
import string

dataset = pd.read_csv('dataset/data.csv')
tempo = np.array(dataset[['text','label_num']])

X = np.array(tempo[:,0])
y = np.array(tempo[:,-1])

#lower case
#removing Punctuation
transTable = str.maketrans(dict.fromkeys(string.punctuation))
X = np.array([sen.lower().translate(transTable) for sen in X])

import re
def remove_hyperlink(word):
    return re.sub(r"http\S+", "", word)

X = np.array([remove_hyperlink(sen) for sen in X])


max_feature = 50000
from keras.preprocessing.text import Tokenizer
tokenizer = Tokenizer(num_words=max_feature)
tokenizer.fit_on_texts(X)
X_features =  np.array(tokenizer.texts_to_sequences(X))

max_len = 0;
for arr in X_features:
    max_len = max(max_len,len(arr))

from keras.preprocessing.sequence import pad_sequences
X_features = pad_sequences(X_features,maxlen=max_len,padding='post')

from sklearn.model_selection import train_test_split as tts

X_train, X_test, y_train, y_test = tts(X_features, y, test_size=0.33, random_state=42,shuffle=True)
X_test, X_valid, y_test, y_valid = tts(X_test, y_test, test_size=0.5, random_state=42,shuffle=True)

#Defining Model

import tensorflow as tf
from tensorflow.keras.layers import Dense,LSTM, Embedding, Dropout, Activation, Bidirectional
#size of the output vector from each layer
embedding_vector_length = 32
#Creating a sequential model
model = tf.keras.Sequential()
#Creating an embedding layer to vectorize
model.add(Embedding(max_feature, embedding_vector_length, input_length=max_len))
#Addding Bi-directional LSTM
model.add(Bidirectional(tf.keras.layers.LSTM(64)))
#Relu allows converging quickly and allows backpropagation
model.add(Dense(16, activation='relu'))
#Deep Learninng models can be overfit easily, to avoid this, we add randomization using drop out
model.add(Dropout(0.1))
#Adding sigmoid activation function to normalize the output
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())




history = model.fit(X_train,y_train,batch_size = 512,epochs=20,validation_data=(X_valid,y_valid))






















