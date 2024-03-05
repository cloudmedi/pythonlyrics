import itertools
import torch
from pytorch_pretrained_bert import BertTokenizer, BertForNextSentencePrediction
import pronouncing
import re


tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = BertForNextSentencePrediction.from_pretrained('bert-large-uncased')
model.eval()

sentence = '[CLS] Charles is a tailor [SEP] He is tall [SEP]'
# sentence = '[CLS] Charles is a tailor [SEP] Excavation is important [SEP]'
toks = tokenizer.tokenize(sentence)
tok_ids = tokenizer.convert_tokens_to_ids(toks)
tok_tensor = torch.LongTensor([tok_ids])
token_type_ids_tensor = torch.LongTensor([[0, 0, 0, 0, 0, 0, 1, 1, 1, 1]])
seq_relationship_logits = model(tok_tensor, token_type_ids_tensor)
print(seq_relationship_logits)


def predict_next_sentence(sentenceA, sentenceBs, tokenizer, model, rhyme=False):
    # Öncelikle, sentenceA ve sentenceBs arasındaki ilişkiyi tahminlemek için BERT modelini kullanıyoruz.
    seq_relationship_logits = get_next_sentence_logits(sentenceA, sentenceBs,
                                                       tokenizer, model)

    # Eğer rhyme=True ise, uyak durumunu kontrol etmek için son kelimeler arasındaki uyakları kontrol ediyoruz.
    if rhyme:
        for i, sentenceB in enumerate(sentenceBs):
            last_word_A = re.sub(r'[^\w]', '', sentenceA.split()[
                -1])  # sentenceA'nın son kelimesini al
            last_word_B = re.sub(r'[^\w]', '', sentenceB.split()[
                -1])  # sentenceB'nin son kelimesini al
            if last_word_A in pronouncing.rhymes(last_word_B) or \
                    last_word_B in pronouncing.rhymes(last_word_A):
                # Eğer son kelimeler arasında uyak varsa, ilgili cümleyi daha yüksek bir puanla işaretle
                seq_relationship_logits[i, 0] += 10

    # Tahmin edilen cümleyi seçmek için en yüksek puanlı cümleyi belirliyoruz.
    predicted_index = seq_relationship_logits[:, 0].argmax().tolist()
    return sentenceBs[predicted_index]


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


def reconstruct_song(lines, tokenizer, model):
    pad_total_size = 2 * max(len(line) for line in lines)
    for ordering in itertools.permutations(lines):
        pair_ids, pair_types, pair_attentions = [], [], []
        for i in range(len(ordering) - 1):
            pair_id, pair_type, pair_attention = \
                get_ids_types_attention_from_sentence_pair(ordering[i],
                                                           ordering[i + 1],
                                                           pad_total_size,
                                                           tokenizer)
            pair_ids.append(pair_id)
            pair_types.append(pair_type)
            pair_attentions.append(pair_attention)
        ids_tensor = torch.LongTensor(pair_ids)
        types_tensor = torch.LongTensor(pair_types)
        attention_tensor = torch.LongTensor(pair_attentions)
        seq_relationship_logits = model(ids_tensor, types_tensor,
                                        attention_tensor)


#         print(seq_relationship_logits)
#         print(sum(seq_relationship_logits[:, 0].tolist()))
#         print(ordering)


def get_next_sentence_logits(sentenceA, sentenceBs, tokenizer, model):
    sentenceA_toks = tokenizer.tokenize(sentenceA)
    sentenceA_ids = tokenizer.convert_tokens_to_ids(sentenceA_toks)
    sentenceA_types = [0] * len(sentenceA_ids)
    sentenceA_attention = [1] * len(sentenceA_ids)
    tok_ids = []
    tok_types = []
    tok_attention = []

    sentenceBs_ids = []
    for sentenceB in sentenceBs:
        sentenceB_toks = tokenizer.tokenize(sentenceB)
        sentenceB_ids = tokenizer.convert_tokens_to_ids(sentenceB_toks)
        sentenceBs_ids.append(sentenceB_ids)

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


tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
model = BertForNextSentencePrediction.from_pretrained('bert-large-uncased')
model.eval()

# Demonstration of predict_next_sentence
predicted_sentence = predict_next_sentence('[CLS] Charles is a tailor [SEP]', ['He is green [SEP]', 'He is very tall [SEP]', 'Excavation is important [SEP]'], tokenizer, model)
print(predicted_sentence)

import csv
import random
import argparse

num_to_generate = 10
num_options = 30


def generate_predictions(args):
    all_lines = []
    all_pairs = []
    with open(args.datafile, encoding='utf8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            lines = row['lyrics'].split('\n')
            for i in range(len(lines) - 1):
                all_pairs.append((lines[i], lines[i + 1]))
                all_lines.append(lines[i])
            all_lines.append(lines[len(lines) - 1])

    sampled_data_x = {}
    sampled_data_y = {}
    correct_pairs = random.sample(all_pairs, num_to_generate)
    for line1, line2 in correct_pairs:
        sampled_data_y[line1] = line2
        sampled_data_x[line1] = [line2]
        sampled_data_x[line1].extend(random.sample(all_lines, num_options - 1))

    #     Batch make predictions to speed up runtime - crashes Colab for using too much RAM
    #
    #     all_sentence_ids, all_sentence_types, all_sentence_attentions = [], [], []
    #     for i, (line1, line2s) in enumerate(sampled_data_x.items()):
    #         sentenceA = line1
    #         for sentenceB in line2s:
    #             sentence_ids, sentence_types, sentence_attentions = \
    #                 get_ids_types_attention_from_sentence_pair(sentenceA, sentenceB, 200, tokenizer)
    #             all_sentence_ids.append(sentence_ids)
    #             all_sentence_types.append(sentence_types)
    #             all_sentence_attentions.append(sentence_attentions)
    #     ids_tensor = torch.LongTensor(all_sentence_ids)
    #     types_tensor = torch.LongTensor(all_sentence_types)
    #     attention_tensor = torch.LongTensor(all_sentence_attentions)
    #     seq_relationship_logits = model(ids_tensor, types_tensor, attention_tensor)
    #     predictions = []
    #     for i, (line1, line2s) in enumerate(sampled_data_x.items()):
    #         prediction_inx = seq_relationship_logits[i*num_options : (i + 1)*num_options, 0].argmax().tolist()
    #         predictions.append((line1, line2s[prediction_inx]))

    with open('predfile_norhyme', 'w') as file_norhyme:
        with open('predfile_rhyme', 'w') as file_rhyme:
            for i, (line1, line2s) in enumerate(sampled_data_x.items()):
                line2 = predict_next_sentence(line1, line2s, tokenizer, model)
                file_norhyme.write(f'{line1}\t{line2}\n')
                line2 = predict_next_sentence(line1, line2s, tokenizer, model,
                                              rhyme=True)
                file_rhyme.write(f'{line1}\t{line2}\n')
                if (i + 1) % 10 == 0:
                    print(f'Finished predicting {i + 1} lines...')
    with open('goldfile', 'w') as file:
        for line1, line2 in sampled_data_y.items():
            file.write(f'{line1}\t{line2}\n')


import csv
import random
import argparse


def generate_predictions_one_song(args):
    num_songs = 10

    with open('goldfile', 'w') as file_gold:
        with open('predfile_random_onesong', 'w') as file_random:
            with open('predfile_norhyme_onesong', 'w') as file_norhyme:
                with open('predfile_rhyme_onesong', 'w') as file_rhyme:
                    print('Deleting old files...')

    for i in range(num_songs):
        chosen_row = None
        n = 1
        with open(args.datafile, encoding='utf8') as csv_file:
            csv_reader = csv.DictReader(csv_file)
            for row in csv_reader:
                if len(row['lyrics'].split('\n')) > 20:
                    continue
                if random.random() < 1 / n:
                    chosen_row = row
                n += 1

        lines = chosen_row['lyrics'].split('\n')

        print(f"Chosen Song: {chosen_row['song']}")
        print()
        print('Lyrics:')
        print('\n'.join(lines))

        sampled_data_x = {}
        sampled_data_y = {}
        for i in range(len(lines) - 1):
            sampled_data_y[lines[i]] = lines[i + 1]
            sampled_data_x[lines[i]] = list(
                set(line for line in lines if line != lines[i]))

        with open('goldfile', 'a') as file_gold:
            with open('predfile_random_onesong', 'a') as file_random:
                with open('predfile_norhyme_onesong', 'a') as file_norhyme:
                    with open('predfile_rhyme_onesong', 'a') as file_rhyme:
                        for i, (line1, line2s) in enumerate(
                                sampled_data_x.items()):
                            line2 = sampled_data_y[line1]
                            file_gold.write(f'{line1}\t{line2}\n')
                            line2 = random.choice(line2s)
                            file_random.write(f'{line1}\t{line2}\n')
                            line2 = predict_next_sentence(line1, line2s,
                                                          tokenizer, model)
                            file_norhyme.write(f'{line1}\t{line2}\n')
                            line2 = predict_next_sentence(line1, line2s,
                                                          tokenizer, model,
                                                          rhyme=True)
                            file_rhyme.write(f'{line1}\t{line2}\n')
#                             if (i + 1) % 10 == 0:
#                                 print(f'Finished predicting {i + 1} lines...')


def generate_song(args):
    num_lines = args.num_lines
    num_choices = args.num_choices

    all_lines = []
    with open(args.datafile, encoding='utf8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            lines = row['lyrics'].split('\n')
            for i in range(len(lines)):
                all_lines.append(lines[i])

    song_lines = [random.choice(all_lines)]

    for i in range(num_lines - 1):
        lines = random.sample(all_lines, num_choices)
        next_line = predict_next_sentence(song_lines[-1], lines, tokenizer,
                                          model, rhyme=True)
        song_lines.append(next_line)

    with open('generate_song3.txt', 'w') as file_song:
        for line in song_lines:
            file_song.write(f'{line}\n')

parser = argparse.ArgumentParser()
parser.add_argument('--datafile', type=str, required=True)
parser.add_argument('--num-lines', type=int, required=True)
parser.add_argument('--num-choices', type=int, required=True)
args = parser.parse_args(['--datafile', 'english_rock.csv', '--num-lines', '10', '--num-choices', '1'])
generate_song(args)