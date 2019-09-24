#!/usr/bin/env python
print "loading modules"
import argparse
import nltk
from gensim.models import Word2Vec
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import cluster
from sklearn import metrics
from sklearn.cluster import KMeans
from nltk.cluster import KMeansClusterer
import numpy as np
import re
import string
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

#stop_words = list(set(nltk.corpus.stopwords.words('english')))
NUM_CLUSTERS=2

# def cluster_by_syntax():
#     return

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

    kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS)
    kmeans.fit(X)

    labels = kmeans.labels_
    centroids = kmeans.cluster_centers_
    #
    # print ("Cluster id labels for inputted data")
    # print (labels)
    # print ("Centroids data")
    # print (centroids)
    #
    # print ("Score (Opposite of the value of X on the K-means objective which is Sum of distances of samples to their closest cluster center):")
    # print (kmeans.score(X))

    silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')
    model = TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y=model.fit_transform(X)
    plt.scatter(Y[:, 0], Y[:, 1], c=assigned_clusters, s=290,alpha=.5)


    for j in range(len(sentences)):
       plt.annotate(assigned_clusters[j],xy=(Y[j][0], Y[j][1]),xytext=(0,0),textcoords='offset points')
       print ("%s %s" % (assigned_clusters[j],  sentences[j]))

    plt.show()


def cluster_by_tfidf(transcript):
    print "clustering by hypernym"
    hypernyms = get_hypernyms(transcript)

    hypes_list = [x['hypernyms'] for x in hypernyms]
    joined_lists = [' '.join(l) for l in hypes_list]

    tfidf_vectorizer = TfidfVectorizer()
    tfidf = tfidf_vectorizer.fit_transform(joined_lists)

    tdidf_kmeans = KMeans(n_clusters=NUM_CLUSTERS).fit(tfidf)

    return tdidf_kmeans


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

    # kmeans_tdidf = cluster_by_tfidf(transcript)
    kmeans_sentiment = cluster_by_sentiment(transcript)




## tfidf special thanks to https://medium.com/@MSalnikov/text-clustering-with-k-means-and-tf-idf-f099bcf95183


## sentiment: https://ai.intelligentonlinetools.com/ml/tag/k-means-clustering-example/




from gensim.models import Word2Vec

from nltk.cluster import KMeansClusterer
import nltk
import numpy as np

from sklearn import cluster
from sklearn import metrics

# training data

# sentences = [['this', 'is', 'the', 'one','good', 'machine', 'learning', 'book'],
#             ['this', 'is',  'another', 'book'],
#             ['one', 'more', 'book'],
#             ['weather', 'rain', 'snow'],
#             ['yesterday', 'weather', 'snow'],
#             ['forecast', 'tomorrow', 'rain', 'snow'],
#             ['this', 'is', 'the', 'new', 'post'],
#             ['this', 'is', 'about', 'more', 'machine', 'learning', 'post'],
#             ['and', 'this', 'is', 'the', 'one', 'last', 'post', 'book']]



# model = Word2Vec(sentences, min_count=1)
#
#
# def sent_vectorizer(sent, model):
#     sent_vec =[]
#     numw = 0
#     for w in sent:
#         try:
#             if numw == 0:
#                 sent_vec = model[w]
#             else:
#                 sent_vec = np.add(sent_vec, model[w])
#             numw+=1
#         except:
#             pass
#
#     return np.asarray(sent_vec) / numw
#
#
# X=[]
# for sentence in sentences:
#     X.append(sent_vectorizer(sentence, model))
#
# print ("========================")
# print (X)
#
#
#
#
# # note with some version you would need use this (without wv)
# #  model[model.vocab]
# print (model[model.wv.vocab])
#
#
#
#
# print (model.similarity('post', 'book'))
# print (model.most_similar(positive=['machine'], negative=[], topn=2))
#
#
#

#
# NUM_CLUSTERS=2
# kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
# assigned_clusters = kclusterer.cluster(X, assign_clusters=True)
# print (assigned_clusters)



# for index, sentence in enumerate(sentences):
#     print (str(assigned_clusters[index]) + ":" + str(sentence))




# kmeans = cluster.KMeans(n_clusters=NUM_CLUSTERS)
# kmeans.fit(X)
#
# labels = kmeans.labels_
# centroids = kmeans.cluster_centers_
#
# print ("Cluster id labels for inputted data")
# print (labels)
# print ("Centroids data")
# print (centroids)
#
# print ("Score (Opposite of the value of X on the K-means objective which is Sum of distances of samples to their closest cluster center):")
# print (kmeans.score(X))
#
# silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')
#
# print ("Silhouette_score: ")
# print (silhouette_score)
#
#
# import matplotlib.pyplot as plt
#
# from sklearn.manifold import TSNE
#
# model = TSNE(n_components=2, random_state=0)
# np.set_printoptions(suppress=True)
#
# Y=model.fit_transform(X)
#
#
# plt.scatter(Y[:, 0], Y[:, 1], c=assigned_clusters, s=290,alpha=.5)
#
#
# for j in range(len(sentences)):
#    plt.annotate(assigned_clusters[j],xy=(Y[j][0], Y[j][1]),xytext=(0,0),textcoords='offset points')
#    print ("%s %s" % (assigned_clusters[j],  sentences[j]))
#
#
# plt.show()
