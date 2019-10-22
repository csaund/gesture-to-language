#!/usr/bin/env python
print "loading modules"

## Standard Fare
import argparse
import json
import numpy as np
import re
import string
import matplotlib.pyplot as plt

## Word Stuff
import nltk
from nltk.cluster import KMeansClusterer
from gensim.models import Word2Vec

## SKLEARN
# from sklearn import cluster
# from sklearn import metrics
# from sklearn.cluster import KMeans
# from sklearn.manifold import TSNE
# from sklearn.feature_extraction.text import TfidfVectorizer

#stop_words = list(set(nltk.corpus.stopwords.words('english')))
NUM_CLUSTERS=2

## def need to keep the sentence with the id.
def get_hypernyms(transcript):
    gesture_hypernyms = []
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

def get_sentences(transcript):
    sentences = []
    for phrase in transcript:
        for gesture in phrase["gestures"]:
            sentences.append(gesture['transcript'])
    return sentences

def get_sentence_sentiment_vector(sent, sentiment_model):
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


def cluster_by_sentiment(transcript):
    print "clustering by sentiment"
    sentences = get_sentences(transcript)
    sentiment_model = Word2Vec(sentences, min_count=1)
    X=[]
    for sentence in sentences:
        X.append(get_sentence_sentiment_vector(sentence, sentiment_model))

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


def cluster_by_tfidf(transcript):
    print "clustering by hypernym"
    hypernyms = get_hypernyms(transcript)

    hypes_list = [x['hypernyms'] for x in hypernyms]
    joined_lists = [' '.join(l) for l in hypes_list]

    tfidf_vectorizer = TfidfVectorizer()
    tfidf = tfidf_vectorizer.fit_transform(joined_lists)

    tdidf_kmeans = KMeans(n_clusters=NUM_CLUSTERS).fit(tfidf)

    return tdidf_kmeans


def cluster_by_syntax(transcript):
    # need to get n-grams
    # then cluster by n-grams I guess??
    return

def input_transcript(transcripts_path):
    transcript = {}
    with open(transcripts_path) as f:
        transcript = json.load(f)
    return transcript

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'path', help='Long mp4 file to be segmented into gestures')
    args = parser.parse_args()

    transcripts_path = './' + args.path + '/' + args.path + '_transcripts_analyzed.json'
    transcript = input_transcript(transcripts_path)

    kmeans_tdidf = cluster_by_tfidf(transcript)
    kmeans_sentiment = cluster_by_sentiment(transcript)
    # plt.show()



## tfidf special thanks to https://medium.com/@MSalnikov/text-clustering-with-k-means-and-tf-idf-f099bcf95183
## sentiment: https://ai.intelligentonlinetools.com/ml/tag/k-means-clustering-example/
