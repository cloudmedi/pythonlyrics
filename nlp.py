import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from textblob import TextBlob


def nlp_analysis(text):
    # Metin içerisindeki kelimeleri ve cümleleri ayırma
    words = word_tokenize(text)
    sentences = sent_tokenize(text)

    # Metin içerisindeki kelimelerin anlamlarını çıkartma
    word_analysis = {}
    for word in words:
        # NLP analizi için textblob kullanımı
        word_analysis[word] = TextBlob(word).definitions

    # Metin içerisindeki cümlelerin anlamlarını çıkartma
    sentence_analysis = {}
    for sentence in sentences:
        # NLP analizi için textblob kullanımı
        sentence_analysis[sentence] = TextBlob(sentence).definitions

    return word_analysis, sentence_analysis


# Örnek bir metin
text = "Her yıl onlarca farklı ülkeden turist ağırlayan bir memleket."
word_analysis, sentence_analysis = nlp_analysis(text)

print("Kelime Anlamları:")
print(word_analysis)

print("\nCümle Anlamları:")
print(sentence_analysis)
