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

    # TODO make this for 3D -- include compound
    def assign_clusters(self, n_clusters=3):
        xy = [(s['sent_score']['pos'], s['sent_score']['neg']) for s in self.sent_data]
        XY = np.array(xy)
        self.XY = XY
        labels = self.clusterer.do_cluster(XY, n_clusters=n_clusters)

        # assign the cluster labels back to the sentiment data we will send back
        for i in range(len(self.sent_data)):  # this had better be the same as labels
            self.sent_data[i]['sent_cluster_id'] = labels[i]

        return labels

    def get_transcript_by_id(self, p_id):
        trans = [p['transcript'] for p in self.sent_data if p['id'] == p_id]
        t = trans[0] if trans else None
        return t

    def get_gesture_ids_by_cluster_id(self, c_id):
        return [p['id'] for p in self.sent_data if p['sent_cluster_id'] == c_id]

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

    # TODO plot abs value of sentiment vs some gesture feature?
    # {id, abs_sent: |pos| + |neg|, feature_vec[x,x,x,x,x,x,x,x,x,x,x]
    # 0 self._palm_vert(gesture, 'l'),
    # 1 self._palm_horiz(gesture, 'l'),
    # 2 self._palm_vert(gesture, 'r'),
    # 3 self._palm_horiz(gesture, 'r'),
    # 4 self._max_hands_apart(gesture),
    # 5 self._min_hands_together(gesture),
    # 6 self._wrists_up(gesture, 'r'),
    # 7 self._wrists_up(gesture, 'l'),
    # 8 self._wrists_down(gesture, 'r'),
    # 9 self._wrists_down(gesture, 'l'),
    # 10 self._wrists_outward(gesture),
    # 11 self._wrists_inward(gesture),
    # 12 self._max_wrist_velocity(gesture, 'r'),
    # 13 self._max_wrist_velocity(gesture, 'l')
    # TODO MOVE THIS!!!!!!!!!!!!!!
    # or don't and add the coloring from the sentiment cluster!!
    def plot_vs_movement(GSM):
        dat = []
        xs = []
        ys = []
        for p in SA.sent_data:
            total_sent = (abs(p['sent_score']['pos'])+abs(p['sent_score']['neg']))
            feature_vec = get_feature_vector_by_gesture_id(GSM.GestureClusterer, p['id'])
            # testing stuff out here
            max_vel = max(feature_vec)
            dat.append((total_sent, max_vel))
            xs.append(total_sent)
            ys.append(max_vel)
        # get feature vectors for all of these ids.
        plt.scatter(xs, ys)
        plt.title('Scatter plot')
        plt.xlabel('total sentiment')
        plt.ylabel('max velocity')
        plt.show()

    # TODO this should DEFINITELY live in GSM
    def get_movement_feature_vecs(self, GSM):
        for p in self.sent_data:
            p['feature_vec'] = GSM.GestureClusterer.get_feature_vector_by_gesture_id(p['id'])

    def scatterplot(xs, ys, xlab="x", ylab="y"):
        plt.scatter(xs, ys)
        plt.title('Scatter plot')
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.show()