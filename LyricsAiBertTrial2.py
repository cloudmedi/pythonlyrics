import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential
import numpy as np
import torch
from pytorch_pretrained_bert import BertTokenizer, BertForNextSentencePrediction
import pronouncing


# PyTorch modeli için gerekli işlevleri tanımlayalım
def predict_next_sentence_pytorch(sentenceA, num_predictions, tokenizer, model):
    tokenized_text_A = tokenizer.tokenize(sentenceA)
    indexed_tokens_A = tokenizer.convert_tokens_to_ids(tokenized_text_A)

    predictions = []
    for sentenceB in range(num_predictions):
        tokenized_text_B = tokenizer.tokenize(sentenceBs[sentenceB])
        indexed_tokens_B = tokenizer.convert_tokens_to_ids(tokenized_text_B)

        # PyTorch modeline giriş verisi hazırlama
        indexed_tokens = indexed_tokens_A + indexed_tokens_B
        segments_ids = [0] * len(indexed_tokens_A) + [1] * len(indexed_tokens_B)
        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])

        # Modelden sonraki cümle tahmini alınması
        with torch.no_grad():
            outputs = model(tokens_tensor, segments_tensors)
            seq_relationship_logits = outputs[0]

        # Tahminlerin eklenmesi
        predictions.append(seq_relationship_logits[0][0].item())

    # En yüksek tahmini döndürme
    max_index = predictions.index(max(predictions))
    return sentenceBs[max_index]


# TensorFlow modeli için gerekli işlevleri tanımlayalım
def generate_lyrics_with_rhyme_and_meter_tensorflow(seed_text, num_lines=12,
                                                    words_per_line=8,
                                                    rhyme_scheme='AABB'):
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
                    if t < maxlen:
                        x_pred[0, t, char_indices[char]] = 1.
            preds = model.predict(x_pred, verbose=0)[0]
            next_index = np.random.choice(len(chars), p=preds)
            next_char = indices_char[next_index]
            seed_text += next_char
            seed_text = seed_text[1:]
            line += next_char
        lines.append(line)

    # Rhyme scheme uygunluğu kontrolü ve uyarlanması
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


def generate_song_lyrics(seed_text, num_lines=12, words_per_line=8,
                         rhyme_scheme='AABB'):
    # PyTorch modelinden kelime ve cümle oluşturma
    generated_lyrics_pytorch = predict_next_sentence_pytorch(seed_text,
                                                             num_lines * words_per_line,
                                                             tokenizer_pytorch,
                                                             model_pytorch)

    # TensorFlow modelinden uyumlu cümleler oluşturma
    lyrics_with_rhyme_and_meter_tensorflow = generate_lyrics_with_rhyme_and_meter_tensorflow(
        seed_text, num_lines, words_per_line, rhyme_scheme)

    # PyTorch ve TensorFlow çıktılarını birleştirme
    combined_lyrics = ""
    for i in range(num_lines):
        combined_lyrics += generated_lyrics_pytorch[
                           i * words_per_line:(i + 1) * words_per_line] + "\n"
        combined_lyrics += lyrics_with_rhyme_and_meter_tensorflow[i] + "\n"

    return combined_lyrics


def generate_song_lyrics(seed_text, num_lines=12, words_per_line=8,
                         rhyme_scheme='AABB'):
    # PyTorch modelinden kelime ve cümle oluşturma
    generated_lyrics_pytorch = predict_next_sentence_pytorch(seed_text,
                                                             num_lines * words_per_line,
                                                             tokenizer_pytorch,
                                                             model_pytorch)

    # TensorFlow modelinden uyumlu cümleler oluşturma
    lyrics_with_rhyme_and_meter_tensorflow = generate_lyrics_with_rhyme_and_meter_tensorflow(
        seed_text, num_lines, words_per_line, rhyme_scheme)

    # PyTorch ve TensorFlow çıktılarını birleştirme
    combined_lyrics = ""
    for i in range(num_lines):
        combined_lyrics += generated_lyrics_pytorch[
                           i * words_per_line:(i + 1) * words_per_line] + "\n"
        combined_lyrics += lyrics_with_rhyme_and_meter_tensorflow[i] + "\n"

    return combined_lyrics

seed_text = "Gökyüzünde parlayan yıldızlar"
num_lines = 8
words_per_line = 6
rhyme_scheme = 'AABB'

generated_poem = generate_song_lyrics(seed_text, num_lines, words_per_line, rhyme_scheme)
print(generated_poem)

import itertools
import torch
from pytorch_pretrained_bert import BertTokenizer, BertForNextSentencePrediction
import pronouncing
import re
import numpy as np
from tensorflow.keras.models import load_model


def predict_next_sentence(sentenceA, sentenceBs, tokenizer, model, rhyme=False):
    seq_relationship_logits = get_next_sentence_logits(sentenceA, sentenceBs,
                                                       tokenizer, model)
    if rhyme:
        for i, sentenceB in enumerate(sentenceBs):
            last_word_A = re.sub(r'[^\w]', '', sentenceA.split()[-1])
            last_word_B = re.sub(r'[^\w]', '', sentenceB.split()[-1])
            if last_word_A in pronouncing.rhymes(
                    last_word_B) or last_word_B in pronouncing.rhymes(
                    last_word_A):
                print(f'{last_word_A} rhymes with {last_word_B}')
                seq_relationship_logits[i, 0] += 10
    return sentenceBs[seq_relationship_logits[:, 0].argmax().tolist()]


def get_ids_types_attention_from_sentence_pair(sentenceA, sentenceB,
                                               pad_total_size, tokenizer):
    sentenceA_ids = tokenizer.convert_tokens_to_ids(
        tokenizer.tokenize(sentenceA))
    sentenceB_ids = tokenizer.convert_tokens_to_ids(
        tokenizer.tokenize(sentenceB))
    padding_size = pad_total_size - len(sentenceA_ids) - len(sentenceB_ids)
    sentence_ids = sentenceA_ids + sentenceB_ids + [0] * padding_size
    sentence_types = [0] * len(sentenceA_ids) + [1] * len(sentenceB_ids) + [
        0] * padding_size
    sentence_attention = [1] * (len(sentenceA_ids) + len(sentenceB_ids)) + [
        0] * padding_size
    return sentence_ids, sentence_types, sentence_attention


def get_next_sentence_logits(sentenceA, sentenceBs, tokenizer, model):
    sentenceA_toks = tokenizer.tokenize(sentenceA)
    sentenceA_ids = tokenizer.convert_tokens_to_ids(sentenceA_toks)
    sentenceA_types = [0] * len(sentenceA_ids)
    sentenceA_attention = [1] * len(sentenceA_ids)
    tok_ids, tok_types, tok_attention = [], [], []

    sentenceBs_ids = [
        tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sentenceB)) for
        sentenceB in sentenceBs]
    max_sentenceB_length = max(
        len(sentenceB_ids) for sentenceB_ids in sentenceBs_ids)

    for sentenceB_ids in sentenceBs_ids:
        padding_size = max_sentenceB_length - len(sentenceB_ids)
        padded_sentenceB_ids = sentenceB_ids + [0] * padding_size
        padded_sentenceB_types = [1] * max_sentenceB_length
        padded_sentenceB_attention = [1] * len(sentenceB_ids) + [
            0] * padding_size
        tok_ids.append(sentenceA_ids + padded_sentenceB_ids)
        tok_types.append(sentenceA_types + padded_sentenceB_types)
        tok_attention.append(sentenceA_attention + padded_sentenceB_attention)

    tok_ids_tensor = torch.LongTensor(tok_ids)
    tok_types_tensor = torch.LongTensor(tok_types)
    tok_attention_tensor = torch.LongTensor(tok_attention)

    seq_relationship_logits = model(tok_ids_tensor, tok_types_tensor,
                                    tok_attention_tensor)
    return seq_relationship_logits


def generate_song_lyrics(seed_text, num_lines=10, words_per_line=8,
                         rhyme_scheme='AABB'):
    lines = [seed_text]
    for _ in range(num_lines - 1):
        lines.append(predict_next_sentence(lines[-1], lines, tokenizer, model,
                                           rhyme=True))
    return '\n'.join(lines)


def load_bert_model():
    tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    model = BertForNextSentencePrediction.from_pretrained('bert-large-uncased')
    model.eval()
    return tokenizer, model


def load_lstm_model():
    loaded_model = load_model('geri_donusumlu_model.h5')
    return loaded_model


# Main function to generate lyrics using both models
def generate_lyrics(seed_text, num_lines=10, words_per_line=8,
                    rhyme_scheme='AABB'):
    # Load models
    tokenizer, bert_model = load_bert_model()
    lstm_model = load_lstm_model()

    # Generate lyrics using BERT and rhyme
    bert_lyrics = generate_song_lyrics(seed_text, num_lines, words_per_line,
                                       rhyme_scheme)

    # Generate lyrics using LSTM
    lstm_lyrics = generate_lyrics_with_rhyme_and_meter(seed_text, num_lines,
                                                       words_per_line,
                                                       rhyme_scheme)

    # Combine lyrics from both models
    combined_lyrics = combine_lyrics(bert_lyrics, lstm_lyrics)

    return combined_lyrics


def combine_lyrics(lyrics1, lyrics2):
    # Split lyrics into lines
    lines1 = lyrics1.split('\n')
    lines2 = lyrics2.split('\n')

    # Combine lines from both sets of lyrics
    combined_lines = [line1 + ' ' + line2 for line1, line2 in
                      zip(lines1, lines2)]

    # Join combined lines into a single string
    combined_lyrics = '\n'.join(combined_lines)

    return combined_lyrics


# Generate lyrics using both models
seed_text = "Gökyüzünde parlayan yıldızlar"
num_lines = 8
words_per_line = 6
rhyme_scheme = 'AABB'

generated_lyrics = generate_lyrics(seed_text, num_lines, words_per_line,
                                   rhyme_scheme)
print(generated_lyrics)
