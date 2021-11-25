from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
import spacy
import re

# Stoppwörter und Lemmatizer laden
german_stopwords = stopwords.words('german')
lemmatizer = spacy.load('de_core_news_sm')


# Daten für BERT bereinigen
def BERTpreprocessing(data, model):
    print('Preparing Data')
    BERT_preprocessedData = dict()
    for key, value in data.items():
        BERT_preprocessedData[key] = BERTCleaning(model, value['content'])
    return BERT_preprocessedData


# Daten für BM25 bereinigen
def BMPreprocessing(data):
    BM25_preprocessedData = dict()
    for key, value in data.items():
        BM25_preprocessedData[key] = BM25Cleaning(value['content'])
    return BM25_preprocessedData


# BERT Preprocessing
def BERTCleaning(model, text):
    sentenceDictionary = dict()
    sentences = sent_tokenize(text, language='german')
    for i in range(len(sentences)):
        sentence = sentences[i]
        cleanedSentence = cleanData(sentence)
        sentenceVector = model.encode(cleanedSentence)
        sentenceDictionary[i] = [sentence, cleanedSentence, sentenceVector]
    return sentenceDictionary


# BM25 Preprocessing
def BM25Cleaning(text):
    text = cleanData(text)
    text = removeStopwords(text)
    text = lemmatize(text)
    text = tokenize(text)
    return text


# Tokenization
def tokenize(text):
    return word_tokenize(text, language='german')


# Stopword Removal
def removeStopwords(text):
    return ' '.join([word for word in text.split() if word not in german_stopwords])


# Lemmatization
def lemmatize(text):
    doc = lemmatizer(text)
    return ' '.join([x.lemma_ for x in doc])


# Data Cleaning
def cleanData(text):
    text = text.lower()
    text = re.sub(r"\s+[a-zA-Z]\s+", ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub('[^a-zA-ZäöüÄÖÜß]', ' ', text)
    return text
