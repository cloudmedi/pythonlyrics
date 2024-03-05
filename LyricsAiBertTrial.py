import random
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Model ve tokenizer yükleme
model_name = 'gpt2'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)


def generate_lyric(starting_text='Sana bir şarkı söyleyeyim...', length=50,
                   temperature=0.7):
    input_ids = tokenizer.encode(starting_text, return_tensors='pt')

    # Do sample True olacak şekilde değiştirildi
    output = model.generate(input_ids, max_length=length, temperature=temperature,
                            num_return_sequences=1, do_sample=True)

    generated_lyric = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_lyric

# Lyric oluşturma
starting_text = 'Bulutlar arasında...'
generated_lyric = generate_lyric(starting_text=starting_text, length=100,
                                 temperature=0.9)
print(generated_lyric)



# Örnek lyric oluşturma
starting_text = 'Bulutlar arasında...'
generated_lyric = generate_lyric(starting_text=starting_text, length=100,
                                 temperature=0.9)
print(generated_lyric)
