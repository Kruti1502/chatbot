
import nltk
import numpy as np
nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()

def tokenize(sentences):
    return nltk.word_tokenize(sentences)

def stem(word):
    return stemmer.stem(word.lower())

def bag_of_words(tokenize_sentence,all_words):
    """
sentence = ["hello","how","are","you"]
words = ["hi","hello","I","you","bye","thank","cool"]
bag = [0,1,0,1,0,0,0]
    """
    tokenize_sentence = [stem(w) for w in tokenize_sentence]
    
    bag = np.zeros(len(all_words),dtype=np.float32)
    for idx,w in enumerate(all_words):
        if w in tokenize_sentence:
            bag[idx] = 1.0
    return bag

#sentence = ["hello","how","are","you"]
#words = ["hi","hello","I","you","bye","thank","cool"]
#bag = bag_of_words(sentence, words)
#print(bag)
#a="how long does shipping take?"
#print(a)
#a=tokenize(a)
#print(a)

#word=["Organize","organizes","organizing"]#stem means common words
#stemmed_words=[stem(w) for w in word]
#print(stemmed_words)





