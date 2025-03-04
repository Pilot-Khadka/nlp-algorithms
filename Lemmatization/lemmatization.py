"""
This is a naive rule-based implementation of lemmatization.

"""
import re
from dictionary_lemmas import LEMMAS

def lemmatize_sentence(sentence):
    word_list = sentence.split(" ")
    res = []
    for word in word_list:
        res.append(lemmatize_words(word))
    return res 

def lemmatize_words(word):
    word = word.lower()

    if word in LEMMAS: # dictionary based lemmatization
        return LEMMAS[word]

    # rule base lemmatization
    if word.endswith("ing"):
        return re.sub(r"ing$", "", word)

    if word.endswith("ed"):
        return re.sub(r"ed$", "", word)

    return word

def main():
    text = "Lemmatization is the process of grouping together the different inflected forms of a word so they can be analyzed as a single item."
    print(lemmatize_sentence(text))

main()
