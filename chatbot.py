import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re


# Télécharger punkt et wordnet de NLTK
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')


dialog = open('data/dialogs.txt', 'r').read()
# Tokeniser le texte
sentences = nltk.sent_tokenize(dialog)


def preprocessing_text(text):
    lemmatizer = WordNetLemmatizer()
    text = text.split()
    text = [lemmatizer.lemmatize(word) for word in text if not word in set(stopwords.words('english'))]
    text = ' '.join(text)  
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text= re.sub(r'\s+', ' ', text)
    text= text.lower().strip()
    
    return text

corpus = [preprocessing_text(sentence) for sentence in sentences]

print(corpus)

#Greetings
GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey", "nods", "hi there", "hello", "I am glad! you are talking to me", "good morning", "good evening", "good afternoon", "good day", "good night", "good to see you", "nice to meet you", "pleased to meet you", "how do you do", "how are you", "how are you doing", "how's it going", "howdy", "how's it hanging", "what's happening", "what's new", "what's going on", "what's the news")
GREETING_RESPONSES = ["hi", "hey", "nods", "hi there", "hello", "I am glad! you are talking to me", "good morning", "good evening", "good afternoon", "good day", "good night", "good to see you", "nice to meet you", "pleased to meet you", "how do you do", "how are you", "how are you doing", "how's it going", "howdy", "how's it hanging", "what's happening", "what's new", "what's going on", "what's the news"]

def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)


def answer(user_input):
    bot_response = ""
    sentences.append(user_input)
    # Vectoriseur TF-IDF
    vectorizer = TfidfVectorizer(tokenizer=preprocessing_text)
    tfidf = vectorizer.fit_transform(sentences)
    cos_similarity = cosine_similarity(tfidf[-1], tfidf)
    similar_sentence = cos_similarity.argsort()[0][-2] # Deuxième phrase la plus similaire
    flatten = cos_similarity.flatten() 
    flatten.sort() 
    vector = flatten[-2]
    if vector == 0:
        bot_response = bot_response + "I am sorry! I don't understand you"
        return bot_response
    else:
        bot_response = bot_response + sentences[similar_sentence]
        return bot_response




