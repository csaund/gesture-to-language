#!/usr/bin/env python
print "loading modules"
import argparse
import nltk
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import re
import string

stop_words = list(set(nltk.corpus.stopwords.words('english')))

## def need to keep the sentence with the id.
def get_sentences(transcripts_path):
    print "loading transcript"
    gesture_hypernyms = []
    with open(transcripts_path) as f:
        transcript = json.load(f)
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

def calculate_tfidf(hypernyms):
    print "calculating clusters"
    hypes_list = [x['hypernyms'] for x in hypernyms]
    joined_lists = [' '.join(l) for l in hypes_list]

    tfidf_vectorizer = TfidfVectorizer()
    tfidf = tfidf_vectorizer.fit_transform(joined_lists)

    kmeans = KMeans(n_clusters=2).fit(tfidf)

    print kmeans.predict(tfidf_vectorizer.transform(["change control completely"]))
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'path', help='Long mp4 file to be segmented into gestures')
    args = parser.parse_args()

    transcripts_path = './' + args.path + '/' + args.path + '_transcripts_analyzed.json'

    hypernyms = get_sentences(transcripts_path)
    with_tfidf = calculate_tfidf(hypernyms)



## special thanks to https://medium.com/@MSalnikov/text-clustering-with-k-means-and-tf-idf-f099bcf95183
