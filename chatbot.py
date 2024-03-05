import string
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import wordnet
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Télécharger punkt et wordnet de NLTK
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


dialog = open('data/dialogs.txt', 'r').read()
# Tokeniser le texte
sentences = nltk.sent_tokenize(dialog)

# Lemmatiser le texte
lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    """Mapper le tag POS au premier caractère que lemmatize() accepte"""
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize_text(text):
    return [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in word_tokenize(text)]

#Greetings
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey", "nods", "hi there", "hello", "I am glad! you are talking to me", "good morning", "good evening", "good afternoon", "good day", "good night", "good to see you", "nice to meet you", "pleased to meet you", "how do you do", "how are you", "how are you doing", "how's it going", "howdy", "how's it hanging", "what's happening", "what's new", "what's going on", "what's the news")
GREETING_RESPONSES = ["hi", "hey", "nods", "hi there", "hello", "I am glad! you are talking to me", "good morning", "good evening", "good afternoon", "good day", "good night", "good to see you", "nice to meet you", "pleased to meet you", "how do you do", "how are you", "how are you doing", "how's it going", "howdy", "how's it hanging", "what's happening", "what's new", "what's going on", "what's the news"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

# Vectoriseur TF-IDF
vectorizer = TfidfVectorizer(tokenizer=lemmatize_text, stop_words='english')

# Adapter et transformer le vectoriseur
tfidf_matrix = vectorizer.fit_transform(sentences)

