import operator
import numpy as np
import fitz
import Preprocessing
import Utils


class Model:
    def __init__(self):
        self.data, self.fileIndex = Utils.loadData()
        self.BERTModel = Utils.loadBERTModel()
        self.BERTpData = Preprocessing.BERTpreprocessing(self.data, self.BERTModel)
        self.BM25pData = Preprocessing.BMPreprocessing(self.data)
        self.BM25Model = Utils.loadBM25Model(self.BM25pData)

    # BM25 Vorhersage
    def get_BM25_Prediction(self, query, theta=1.598):
        query = Preprocessing.BM25Cleaning(query.lower())
        doc_scores = self.BM25Model.get_scores(query)
        bm25Scores = dict()
        for i in range(len(doc_scores)):
            if doc_scores[i] > theta:
                bm25Scores[i] = doc_scores[i]
        return dict(sorted(bm25Scores.items(), key=operator.itemgetter(1), reverse=True))

    # BERT Vorhersage
    def get_BERT_Prediction(self, query, theta=0.565):
        queryEncoded = self.BERTModel.encode(query)
        bertScores, allBestSentences = dict(), dict()

        for docID, sentenceDictionary in self.BERTpData.items():
            scores, docBestSentences = [], dict()
            for sentID, values in sentenceDictionary.items():
                similarity = Utils.cosineSimilarity(queryEncoded, values[2])
                docBestSentences[sentID] = similarity
                scores.append(similarity)
            docBestSentences = dict(sorted(docBestSentences.items(), key=operator.itemgetter(1), reverse=True))
            allBestSentences[docID] = docBestSentences
            docSimilarity = np.mean(sorted(scores, reverse=True)[:5])
            if docSimilarity > theta:
                bertScores[docID] = docSimilarity
            if len(scores) > 1:
                bertScores = dict(sorted(bertScores.items(), key=operator.itemgetter(1), reverse=True))
        return bertScores, allBestSentences

    # Kombinierte Vorhersage
    def get_Combined_Prediction(self, query, theta=0.169, k1=0.655):
        bertScores, bestSentences = self.get_BERT_Prediction(query)
        bm25Scores = self.get_BM25_Prediction(query)
        combinedScores = dict()

        if len(bm25Scores) == 0 and len(bertScores) == 0:
            return dict(), bestSentences
        if len(bm25Scores) == 0:
            return bertScores,bestSentences
        if len(bertScores) == 0:
            return bm25Scores, bestSentences

        bm25Scores = Utils.normalize(bm25Scores)
        bertScores = Utils.normalize(bertScores)

        # Kombinieren mit k1 Gewichtung
        for docId in self.BERTpData.keys():
            bm25Score, bertScore = 0, 0
            if int(docId) in bm25Scores:
                bm25Score = bm25Scores[int(docId)]
            if int(docId) in bertScores:
                bertScore = bertScore[int(docId)]
            combined = k1 * bertScore + (1 - k1) * bm25Score
            if combined > theta:
                combinedScores[docId] = k1 * bertScore + (1 - k1) * bm25Score
        resultScores = dict(sorted(combinedScores.items(), key=operator.itemgetter(1), reverse=True))
        return resultScores, bestSentences

    # Relevante Sätze in PDF markieren, Dokument abspeichern und Output formatieren
    def prepareOutput(self, docIDs, bestSentences, numberOfSentences=3):
        directory = 'static/Betriebsvereinbarungen/'
        saveDirectory = directory + 'TempDocs/'
        outputText = ['Das konnte ich zu deiner Anfrage finden.','']
        for i in range(len(docIDs)):

            # Get most relevant Sentences for current Document
            sentenceDict = dict(sorted(bestSentences[str(docIDs[i])].items(), key=operator.itemgetter(1), reverse=True))
            sentenceIDs = list(sentenceDict.keys())[:numberOfSentences]

            # Variable for setting relevant page number
            pageNumber = 0

            # Find and open the document
            doc = fitz.open(directory + self.fileIndex[str(docIDs[i])])
            # Iterate through every relevant sentence
            for sentenceID in sentenceIDs:
                text = self.BERTpData[str(docIDs[i])][sentenceID][0]
                # Check if the page contains the text
                for page in doc:
                    text_instances = page.search_for(text)
                    # Highlight the searched for text
                    for inst in text_instances:
                        # Set pageNumber for the first relevant sentence
                        if pageNumber == 0:
                            pageNumber = page.number + 1
                        # Highlight the text
                        highlight = page.add_highlight_annot(inst)
                        highlight.update()

            # Save and close the file
            outputName = str(i) + '.pdf'
            doc.save(saveDirectory + outputName, garbage=1, deflate=True, clean=True)
            doc.close()

            # Add the file and sentence to the Output-String
            docLocation = saveDirectory + outputName + '#page=' + str(pageNumber)

            docName = self.fileIndex[str(docIDs[i])]
            if len(docName) > 35:
                docName = docName[:32] + '...'
            representativeSentence = self.BERTpData[str(docIDs[i])][sentenceIDs[0]][0]
            if len(representativeSentence) > 80:
                representativeSentence = representativeSentence[:77] + '...'
            hyperlink = '<a href="{}" type="application/pdf" target="_blank">{}</a>'.format(docLocation, docName)
            outputText.extend([hyperlink,representativeSentence,''])

        return "<br/>".join(map(str, outputText))

    # Oberste Methode für die Vorausage anhand einer Nutzeranfrage
    def get_prediction(self, query, maxNumberOfDocuments=5, numberOfSentences=3):
        scores, bestSentences = self.get_Combined_Prediction(query)
        if len(scores) == 0:
            return 'Entschuldigung, ich konnte keine passende Information zu der Anfrage finden.'
        return self.prepareOutput(list(scores.keys())[:maxNumberOfDocuments], bestSentences, numberOfSentences)