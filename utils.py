import nltk
from scipy.sparse.csgraph import yen
nltk.download('punkt_tab')
import numpy as np
import pandas as pd

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import pickle
import random
import json
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
#cargar los datos - #Creo que debe ser un archivo .json
path = '/content/drive/MyDrive/CursoAI/intents.json' #Cada uno con la dirección de la carpeta.
with open(path, 'r', encoding='utf-8') as file:
    data = json.load(file)
#creamos el stemmer
stemmer = PorterStemmer()
#Preprocesamiento
vocab = []
tags = []
patterns = []
labels = []
for intent in data['intents']: #Reemplazar intent con archivo.json que generemos.
    for pattern in intent['patterns']:
        tokens = word_tokenize(pattern.lower())
        stemmed = [stemmer.stem(w) for w in tokens]
        vocab.extend(stemmed)
        patterns.append(stemmed)
        labels.append(intent['tag'])
    if intent['tag'] not in tags:
        tags.append(intent['tag'])
vocab = sorted(set(vocab))

#One-hot input #Variables

X = []
Y = []
#encoder (decodificador)
encoder = LabelEncoder()
encoder_labels = encoder.fit_transform(labels)
for pattern in patterns:
    bag = [1 if word in pattern else 0 for word in vocab]
    X.append(bag)
Y = encoder_labels
#convertir las variables a arreglos de numpy
X = np.array(X)
Y = np.array(Y)
#Modelo
D = len(X[0]) #Cantidad de Entradas
C = len(tags) #Cantidad de Etiquetas
model = Sequential()
#capa de entrada - densa
#Capa densa, con 8 neuronas
model.add(Dense(8, input_shape= (D,), activation = 'relu')) #Coma y vacio (, ), porque se saben los datos de entrada pero no los de salida
#capa densa 2 del modelo
model.add(Dense(8, activation= 'relu')) #Las salidas de la capa densa son los que van a capa densa 2
#capa densa con la cantidad de neuronas que tiene cada categoria
model.add(Dense(C, activation='softmax'))
#compilamos el modelo
model.compile(
    loss='sparse_categorical_crossentropy',
    optimizer ='adam',
    metrics = ['accuracy'],
)
#entrenamos el modelo
model.fit(X, Y, epochs = 200, verbose=0)

#Función para procesar la entrada
def predict_class(text):
    tokens = word_tokenize(text.lower())
    stemmed = [stemmer.stem(w) for w in tokens]
    bag = np.array([1 if word in stemmed else 0 for word in vocab])
    res = model.predict(np.array([bag]), verbose = 0)[0]
    idx = np.argmax(res)
    tag = encoder.inverse_transform([idx])[0]
    return tag
#Función para dar las respuestas
def get_response(tag, context):
    for intent in data['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    return 'Error, vamos de nuevo'