import numpy as np
import json
from rank_bm25 import *
from sentence_transformers import SentenceTransformer
import Preprocessing
import operator

# QWERTZ Tastatur-Layout
keyboard = [
    ['^', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', 'ß', '='],
    ['', 'q', 'w', 'e', 'r', 't', 'z', 'u', 'i', 'o', 'p', 'ü', '+', ],
    ['', 'a', 's', 'd', 'f', 'g', 'h', 'j', 'k', 'l', 'ö', 'ä', '#'],
    ['', '>', 'y', 'x', 'c', 'v', 'b', 'n', 'm', ',', '.', '-']]

# Vordefiniertes Vokabular laden
with open("Daten/Vocab.json", encoding='utf-8') as file:
    vocab = json.load(file)


# Die Daten laden
def loadData():
    with open("Daten/Data.json", encoding='utf-8') as file:
        data = json.load(file)

    with open("Daten/FileIndex.json", encoding='utf-8') as file:
        fileIndex = json.load(file)
    return data, fileIndex


# BM25 Model laden
def loadBM25Model(documents):
    return BM25Okapi(list(documents.values()))


# BERT Model laden
def loadBERTModel():
    print('Loading BERT Model')
    return SentenceTransformer("svalabs/bi-electra-ms-marco-german-uncased")


# Kosinus-Ähnlichkeit berechnen
def cosineSimilarity(vector1, vector2):
    return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))


# Dictionary normalisieren
def normalize(dataDict):
    if len(dataDict) == 1:
        for docId, score in dataDict.items():
            dataDict[docId] = 1
        return dataDict
    else:
        minDataDict, maxDataDict = np.min(list(dataDict.values())), np.max(list(dataDict.values()))
        # DataDict normalisieren
        for docId, score in dataDict.items():
            dataDict[docId] = (score - minDataDict) / (maxDataDict - minDataDict)
        return dataDict

# Rechtschreibungskorrektur anhand Levenshtei-Distanz
def checkSpelling(query):
    queryTokens = Preprocessing.tokenize(query)

    # Durch Anfrage iterieren
    for token in queryTokens:
        if len(token) > 1:
            wordDict = vocab[token[0]]

            # Wenn Token nicht in Vokabular vorkommt
            if token not in wordDict:
                distanzen = dict()

                # Mit jedem Wort mit identischen Anfangsbuchstaben vergleichen
                for word, count in wordDict.items():
                    distanz = levenshteinDistanz(token, word)
                    distanzen[word] = distanz

                # Sortieren und bestes Wort als Basis wählen
                sortedDistances = dict(sorted(distanzen.items(), key=operator.itemgetter(1), reverse=False))
                bestWords, bestDistance = [list(sortedDistances.keys())[0]], list(sortedDistances.values())[0]

                # Wenn mehrere Worte eine identische Distanz besitzen
                for word, editDistance in sortedDistances.items():
                    if editDistance > bestDistance:
                        break
                    else:
                        bestWords.append(word)

                # Bei gleicher Distanz Wort mit höhester Frequenz auswählen
                maxCount, winnerWord = -1, ''
                for word in bestWords:
                    count = vocab[word[0]][word]
                    if count > maxCount:
                        maxCount, winnerWord = count, word

                # Ausgewähltes Wort durch alten Token ersetzen
                query = query.replace(token, winnerWord)

    # Angepasste Anfrage zurückgeben
    return query


# Methode zur Berechnung der Levensthein-Distanz
def levenshteinDistanz(wort1, wort2):
    # Matrix mit Null-Werten initialsieren
    distanzen = np.zeros((len(wort1) + 1, len(wort2) + 1))

    # Erste Spalte mit den richtigen Werten befüllen
    for t1 in range(len(wort1) + 1):
        distanzen[t1][0] = t1

    # Erste Zeile mit den richtigen Werten befüllen
    for t2 in range(len(wort2) + 1):
        distanzen[0][t2] = t2

    # Distanzmatrix mit Werten befüllen
    for p1 in range(1, len(wort1) + 1):
        for p2 in range(1, len(wort2) + 1):
            # Beide Worte haben identisches Zeichen
            if wort1[p1 - 1] == wort2[p2 - 1]:
                distanzen[p1][p2] = distanzen[p1 - 1][p2 - 1]
            else:
                insertionDistance = distanzen[p1][p2 - 1] + 1
                deletionDistance = distanzen[p1 - 1][p2] + 1
                substitutionCost = distanzen[p1 - 1][p2 - 1] + \
                                   euklidischeDistanz(wort1[p1 - 1], wort2[p2 - 1])
                distanzen[p1][p2] = min(insertionDistance, deletionDistance, substitutionCost)
    return distanzen[len(wort1)][len(wort2)]


# Methode zur Lokalisierung der Koordinaten des Zeichens
def getCoordinates(char):
    for r in keyboard:
        if char in r:
            return keyboard.index(r), r.index(char)
    raise ValueError(char + " nicht gefunden")


# Berechnung der euklidischen Distanz
def euklidischeDistanz(char1, char2):
    x1, x2 = getCoordinates(char1), getCoordinates(char2)
    return ((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2) ** 0.5


def printDistances(distances, token1Length, token2Length):
    for t1 in range(token1Length + 1):
        for t2 in range(token2Length + 1):
            print(distances[t1][t2], end=" ")
        print()
