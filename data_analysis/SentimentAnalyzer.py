from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import numpy as np
from Clusterer import *

# To visualize clusters you can do
# ~standard loading GSM stuff (top of GestureSentenceManager.py~
# INCLUDING initializing the "SentenceManager"
# then
# SA = SentimentAnalyzer(GSM.SentenceClusterer.agd['phrases'])
# see sent data thru SA.sent_data    # peep at SA.sent_data[0]
# SA.assign_clusters()
# SA.visualize()     or SA.visualize_multi([2, 3, 5, 8])

# analyzes sentences according to https://github.com/cjhutto/vaderSentiment#python-code-example
# so returns in format:
# {
#   'pos': 0.746,
#   'compound': 0.8316,
#   'neu': 0.254,
#   'neg': 0.0
# }
class SentimentAnalyzer:
    def __init__(self, phrases):
        self.analyzer = SentimentIntensityAnalyzer()
        if len(phrases):
            ps = [p['phase']['transcript'] for p in phrases]
            self.clusterer = Clusterer(ps)
            self.sent_data = self.get_sentiment_data(phrases)
        else:
            self.clusterer = None
            self.sent_data = []
        self.XY = []

    # takes array in form of [{'id': XXX, 'phase':{'transcript': "words to be analyzed"}}]
    def get_sentiment_data(self, phrases):
        sent_data = []          # create the data we will send back
        for phrase in phrases:
            d = {'id': phrase['id'], 'transcript': phrase['phase']['transcript']}
            d['sent_score'] = self.analyze_sentence(d['transcript'])
            sent_data.append(d)

        self.sent_data = sent_data
        return self.sent_data

    def assign_clusters(self, n_clusters=3):
        xy = [(s['sent_score']['pos'], s['sent_score']['neg']) for s in self.sent_data]
        XY = np.array(xy)
        self.XY = XY
        labels = self.clusterer.do_cluster(XY, n_clusters=n_clusters)

        # assign the cluster labels back to the sentiment data we will send back
        for i in range(len(self.sent_data)):  # this had better be the same as labels
            self.sent_data[i]['sent_cluster_id'] = labels[i]

        return labels

    def visualize(self, n_clusters=3):
        self.clusterer.vis_clusters(self.XY, n_clusters)

    def visualize_multi(self, n_cluster_range=[3, 4]):
        self.clusterer.silhouette_comparison(self.XY, n_cluster_range)

    # takes a string that represents a single sentence,
    # returns
    def analyze_sentence(self, s):
        score = self.analyzer.polarity_scores(s)
        return score

    def analyze_sentences(self, s_list):
        scores = []
        for s in s_list:
            scores.append(self.analyzer.polarity_scores(s))
        return scores
