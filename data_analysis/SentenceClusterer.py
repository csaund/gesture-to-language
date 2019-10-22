#!/usr/bin/env python
print "loading modules"

## Standard Fare
import argparse
import json
import numpy as np
import re
import string
import os
import matplotlib.pyplot as plt

## Word Stuff
import nltk
from nltk.cluster import KMeansClusterer
from gensim.models import Word2Vec

from sklearn import cluster
from sklearn import metrics
from sklearn.manifold import TSNE

devKey = str(open("/Users/carolynsaund/devKey", "r").read()).strip()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/carolynsaund/google-creds.json"
from common_helpers import *

#stop_words = list(set(nltk.corpus.stopwords.words('english')))
NUM_CLUSTERS=2


# gets the full transcript straight from the cloud, baby.
class SentenceClusterer():
    def __init__(self, speaker):
        transcript_bucket = "full_timings_with_transcript_bucket"
        speaker_transcript_name = "%s_timings_with_transcript.json" % speaker
        temp_file = "tempfile.json"
        # TODO make all of this more seamless -- make the temp file somewhere else
        # and delete it there too (like in common_helpers.py)
        download_blob(transcript_bucket, speaker_transcript_name, temp_file)
        self.transcript_with_timings = read_data(temp_file)
        os.remove(temp_file)

    ## def need to keep the sentence with the id.
    def get_hypernyms(self, transcript):
        gesture_hypernyms = []
        ## TODO make this pick out the key words in a sentence and then cluster
        for phrase in transcript:
            for gesture in phrase["gestures"]:
                hypernyms = []
                for key in gesture["hypernyms"].keys():
                    hypernyms.append(gesture["hypernyms"][key])
                ## flatten the list of hypernyms so it's kind of a grab bag of words.
                ## this way we can later to tfidf k-means clustering
                hypernyms = [y for x in hypernyms for y in x]
                gest = {"id": gesture["id"],
                        "hypernyms": hypernyms}
                gesture_hypernyms.append(gest)
        return gesture_hypernyms

    def get_sentences(self, transcript=None):
        trans = transcript if transcript else self.transcript_with_timings
        tr = trans['phrases']
        sentences = [g['phase']['transcript'] for g in tr]
        return sentences

    def get_sentence_sentiment_vector(self, sent, sentiment_model):
        sent_vec =[]
        numw = 0
        for w in sent:
            try:
                if numw == 0:
                    sent_vec = sentiment_model[w]
                else:
                    sent_vec = np.add(sent_vec, sentiment_model[w])
                numw+=1
            except:
                pass
        return np.asarray(sent_vec) / numw


    def cluster_by_sentiment(self, transcript=None):
        print "clustering by sentiment"
        trans = transcript if transcript else self.transcript_with_timings
        sentences = self.get_sentences(trans)
        sentences = self._drop_empties(sentences)
        sentiment_model = Word2Vec(sentences, min_count=1)
        X= map(lambda s: self.get_sentence_sentiment_vector(s, sentiment_model), sentences)

        kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
        assigned_clusters = kclusterer.cluster(X, assign_clusters=True)

        kmeans_sentiment = cluster.KMeans(n_clusters=NUM_CLUSTERS)
        kmeans_sentiment.fit(X)

        labels = kmeans_sentiment.labels_
        centroids = kmeans_sentiment.cluster_centers_

        silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')
        model = TSNE(n_components=2, random_state=0)
        np.set_printoptions(suppress=True)
        Y=model.fit_transform(X)
        plt.scatter(Y[:, 0], Y[:, 1], c=assigned_clusters, s=290,alpha=.5)

        for j in range(len(sentences)):
           plt.annotate(assigned_clusters[j],xy=(Y[j][0], Y[j][1]),xytext=(0,0),textcoords='offset points')
           print ("%s %s" % (assigned_clusters[j],  sentences[j]))

        return kmeans_sentiment

    def _drop_empties(self, v):
        print "Dropping %s empty strings." % v.count('')
        return [i for i in v if i != '']

    def cluster_by_tfidf(self, transcript):
        print "clustering by hypernym"
        hypernyms = self.get_hypernyms(transcript)

        hypes_list = [x['hypernyms'] for x in hypernyms]
        joined_lists = [' '.join(l) for l in hypes_list]

        tfidf_vectorizer = TfidfVectorizer()
        tfidf = tfidf_vectorizer.fit_transform(joined_lists)

        tdidf_kmeans = KMeans(n_clusters=NUM_CLUSTERS).fit(tfidf)

        return tdidf_kmeans


    def cluster_by_syntax(self, transcript):
        # need to get n-grams
        # then cluster by n-grams I guess??
        return

    def input_transcript(self, transcripts_path):
        transcript = {}
        with open(transcripts_path) as f:
            transcript = json.load(f)
        return transcript

    # if __name__ == '__main__':
    #     parser = argparse.ArgumentParser(
    #         description=__doc__,
    #         formatter_class=argparse.RawDescriptionHelpFormatter)
    #     parser.add_argument(
    #         'path', help='Long mp4 file to be segmented into gestures')
    #     args = parser.parse_args()
    #
    #     transcripts_path = './' + args.path + '/' + args.path + '_transcripts_analyzed.json'
    #     transcript = input_transcript(transcripts_path)
    #
    #     kmeans_tdidf = cluster_by_tfidf(transcript)
    #     kmeans_sentiment = cluster_by_sentiment(transcript)
        # plt.show()



    ## tfidf special thanks to https://medium.com/@MSalnikov/text-clustering-with-k-means-and-tf-idf-f099bcf95183
    ## sentiment: https://ai.intelligentonlinetools.com/ml/tag/k-means-clustering-example/
