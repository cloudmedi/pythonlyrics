import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
import numpy as np

with open("hande_yener_lyrics.txt", "r", encoding="utf-8") as file:
    lyrics = file.readlines()




# Preprocessing
chars = sorted(list(set(lyrics)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))
maxlen = 40
step = 3
sentences = []
next_chars = []

# Preprocessing
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



# Define the model
model = Sequential([
    LSTM(128, input_shape=(maxlen, len(chars))),
    Dense(len(chars), activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam')

# Train the model
model.fit(x, y, batch_size=20, epochs=30)
model.save('geri_donusumlu_model.h5')


def loadModel():

    loaded_model = load_model('geri_donusumlu_model.h5')


# Generate lyrics
def generate_lyrics(seed_text, num_chars=12):
    generated = seed_text
    for _ in range(num_chars):
        if seed_text not in char_indices:
            seed_text = np.random.choice(list(chars))
        x_pred = np.zeros((1, maxlen, len(chars)))
        for t, char in enumerate(seed_text):
            if char in char_indices:
                if t < maxlen:  # Check if t is within bounds
                    x_pred[0, t, char_indices[char]] = 1.
        preds = model.predict(x_pred, verbose=0)[0]
        next_index = np.random.choice(len(chars), p=preds)
        next_char = indices_char[next_index]
        seed_text += next_char
        seed_text = seed_text[1:]
        generated += next_char
    return generated



import pronouncing


# Function to find rhyming words
def find_rhyme(word):
    return pronouncing.rhymes(word)


# Generate lyrics with rhyme and meter
def generate_lyrics_with_rhyme_and_meter(seed_text, num_lines=8,
                                         words_per_line=8, rhyme_scheme='AABB'):
    generated = seed_text
    lines = []
    for _ in range(num_lines):
        line = seed_text
        for _ in range(words_per_line - 1):
            if seed_text not in char_indices:
                seed_text = np.random.choice(list(chars))
            x_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(seed_text):
                if char in char_indices:
                    if t < maxlen:  # Check if t is within bounds
                        x_pred[0, t, char_indices[char]] = 1.
            preds = model.predict(x_pred, verbose=0)[0]
            next_index = np.random.choice(len(chars), p=preds)
            next_char = indices_char[next_index]
            seed_text += next_char
            seed_text = seed_text[1:]
            line += next_char
        lines.append(line)

    # Ensure lines adhere to rhyme scheme
    if rhyme_scheme == 'AABB':
        for i in range(0, num_lines, 2):
            rhyming_word = lines[i].split()[-1]
            rhyming_words = find_rhyme(rhyming_word)
            if rhyming_words:
                rhyming_line = np.random.choice(rhyming_words)
                if len(rhyming_line.split()) > 1:
                    lines[i + 1] = lines[i + 1].rsplit(' ', 1)[0] + ' ' + \
                                   rhyming_line.split(' ', 1)[1]
    elif rhyme_scheme == 'ABAB':
        for i in range(0, num_lines, 4):
            rhyming_word1 = lines[i].split()[-1]
            rhyming_word2 = lines[i + 2].split()[-1]
            rhyming_words1 = find_rhyme(rhyming_word1)
            rhyming_words2 = find_rhyme(rhyming_word2)
            if rhyming_words1:
                rhyming_line1 = np.random.choice(rhyming_words1)
                if len(rhyming_line1.split()) > 1:
                    lines[i + 1] = lines[i + 1].rsplit(' ', 1)[0] + ' ' + \
                                   rhyming_line1.split(' ', 1)[1]
            if rhyming_words2:
                rhyming_line2 = np.random.choice(rhyming_words2)
                if len(rhyming_line2.split()) > 1:
                    lines[i + 3] = lines[i + 3].rsplit(' ', 1)[0] + ' ' + \
                                   rhyming_line2.split(' ', 1)[1]

    return '\n'.join(lines)


# Generate lyrics with rhyme and meter using your seed text
seed_text = "Kalp kırılır, korkma"
generated_lyrics = generate_lyrics_with_rhyme_and_meter(seed_text)
print(generated_lyrics)

# # Generate lyrics with a seed text
# seed_text = "Kalp kırılır, korkma"
# generated_lyrics = generate_lyrics(seed_text, num_chars=300)
# print(generated_lyrics)

