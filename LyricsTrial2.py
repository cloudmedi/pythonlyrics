from transformers import AutoTokenizer
import os
import re
import tensorflow as tf
from datasets import load_dataset


file_name = "ticaret-yorum.csv"
path = "/content/gdrive/MyDrive/Colab Notebooks/input/"

reviews_dataset = load_dataset("csv", data_files= path+file_name)

reviews_sample = reviews_dataset["train"].shuffle(seed=42).select(range(4000))


def remove_repeated(example):
  example["review"] = example["review"].replace('...', '')
  example["review"] = example["review"].replace(',"', '. ')
  example["review"] = example["review"].replace('!.', '.')
  example["review"] = example["review"].replace('!,', '. ')
  example["review"] = example["review"].replace('"', '')
  example["review"] = re.sub(
    '([a-zA-Z0-9zığüşöçZİĞÜŞÖÇ]),([a-zA-Z0-9zığüşöçZİĞÜŞÖÇ])', '\\1. \\2',
    example["review"])

  return {"review": example["review"].replace('Devamını oku', '')}


reviews_sample = reviews_sample.train_test_split(train_size=0.9, seed=42)
# Rename the default "test" split to "validation"
reviews_sample["validation"] = reviews_sample.pop("test")

reviews_sample


context_length = 40
pretrained_tokenizer = AutoTokenizer.from_pretrained("redrussianarmy/gpt2-turkish-cased")

outputs = pretrained_tokenizer(
    reviews_sample["train"][:2]["review"],
    truncation=True,
    max_length=context_length,
    return_overflowing_tokens=False,
    return_length=True,
)

print(f"Input IDs length: {len(outputs['input_ids'])}")
print(f"Input chunk lengths: {(outputs['length'])}")

print("vocab_size: ", len(pretrained_tokenizer))

txt = "Sürat Kargom Hala Gelmedi,1402 numaralı kargom adatepe şubesinde."
tokens = pretrained_tokenizer(txt)['input_ids']
print(tokens)

converted = pretrained_tokenizer.convert_ids_to_tokens(tokens)
print(converted)


def get_training_corpus():
    batch_size = 1000
    return (
        reviews_sample["train"][i : i + batch_size]["review"]
        for i in range(0, len(reviews_sample["train"]), batch_size)
    )
training_corpus = get_training_corpus()

for reviews in get_training_corpus():
    print(len(reviews))


vocab_size = 52000
tokenizer = pretrained_tokenizer.train_new_from_iterator(training_corpus,vocab_size)

txt = "Sürat Kargom Hala Gelmedi,1402 numaralı kargom adatepe şubesinde."
tokens = tokenizer(txt)['input_ids']
print(tokens)

converted = tokenizer.convert_ids_to_tokens(tokens)
print(converted)

path="./"
file_name="turkishReviews-ds-mini"
tokenizer.save_pretrained(path+file_name)

loaded_tokenizer = AutoTokenizer.from_pretrained("./turkishReviews-ds-mini")

txt = "Sürat Kargom Hala Gelmedi,1402 numaralı kargom adatepe şubesinde."
tokens = tokenizer(txt)['input_ids']
print("trained tokenizer:", tokens)
tokens = loaded_tokenizer(txt)['input_ids']
print("loaded tokenizer:", tokens)
#print(f"Chunk mapping: {outputs['overflow_to_sample_mapping']}")