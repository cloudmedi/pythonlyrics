import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import datetime

# Giriş verileri ile ilgili kısmı okuma
with open("hande_yener_lyrics.txt", "r", encoding="utf-8") as file:
    lyrics = file.readlines()

# Verilerin işlenmesi
chars = sorted(list(set(lyrics)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
maxlen = 40
step = 3
sentences = []
next_chars = []

for i in range(0, len(lyrics) - maxlen, step):
    sentences.append(lyrics[i: i + maxlen])
    next_chars.append(lyrics[i + maxlen])
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=bool)
y = np.zeros((len(sentences), len(chars)), dtype=bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        if char in char_indices:
            x[i, t, char_indices[char]] = 1
    if next_chars[i] in char_indices:
        y[i, char_indices[next_chars[i]]] = 1

# Model tanımlama
model = Sequential([
    LSTM(128, input_shape=(maxlen, len(chars))),
    Dense(len(chars), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam')

# TensorBoard callback'u oluşturma
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Model eğitimi
model.fit(x, y, batch_size=20, epochs=50, callbacks=[tensorboard_callback])

