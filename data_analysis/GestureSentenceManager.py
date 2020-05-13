#!/usr/bin/env pythons
import os
import pandas as pd
from google.cloud import storage
from common_helpers import get_data_from_path, get_data_from_blob, read_data, download_blob, filter_words_by_syntax, filter_words_out_by_syntax
import nltk
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from data_analysis.VideoManager import VideoManager
from data_analysis.GestureClusterer import GestureClusterer
from data_analysis.GestureMovementHelpers import GESTURE_FEATURES
from data_analysis.RhetoricalClusterer import RhetoricalClusterer
from data_analysis.GestureSplicer import GestureSplicer


VERBS = ["V", "VB", "VBD", "VBD", "VBZ", "VBP", "VBN"]
NOUNS = ["NN", "NNP", "NNS"]
ADJ = ["JJ"]

## the following commands assume you have a full transcript in the cloud
## and also all the timings.
# sys.path.extend(['C:/Users/carolyns/PycharmProjects/gesture-to-language/data_analysis'])
# from GestureSentenceManager import *
# exec(open('setup.py').read())
# GSM = GestureSentenceManager("conglomerate_under_10")
# GSM.downsample_speaker()
# GSM._initialize_sentence_clusterer()
# GSM.cluster_gestures()    or    GSM.cluster_gestures_under_n_words(10)
# GSM.test_k_means_gesture_clusters()
# report = GSM.report_clusters()
# GSM.print_sentences_by_cluster(0)
# GSM.cluster_sentences_gesture_independent()    or     GSM.cluster_sentences_gesture_independent_under_n_words(10)
# GSM.assign_gesture_cluster_ids_for_sentence_clusters()
# GSM.combine_all_gesture_data()
# Ann = Analyzer(GSM)           # Before this must run GSM.combine_all_gesture_data()


# from GestureSentenceManager import *
# GSM = GestureSentenceManager("conglomerate_under_10")
# GSM.GestureClusterer.cluster_gestures_disparate_seeds(None, max_cluster_distance=0.03, max_number_clusters=27)
# GSM.cluster_sentences_gesture_independent()

# exec(open('setup.py').read())
# GSM = GestureSentenceManager("test")
# GS = GestureSplicer()
# df = GS.splice_gestures(GSM.df)
# GSM._setup()

class GestureSentenceManager:
    def __init__(self, speaker):
        # this is where the magic is gonna happen.
        # get all the gestures
        self.speaker = speaker
        self.cluster_bucket_name = "%s_clusters" % speaker
        self.full_transcript_bucket = "full_timings_with_transcript_bucket"
        self.gesture_transcript = None
        self.gesture_sentence_clusters = {}
        self.transcript = self.get_transcript()
        self.VideoManager = VideoManager()
        self.df = self.get_df()
        self.GestureSplicer = GestureSplicer()
        # TODO will need to re-initialize these if we use gesture splicing!!
        self.GestureClusterer = GestureClusterer(self.df)
        self.RhetoricalClusterer = RhetoricalClusterer(self.df)

    ################################################
    ##################### SETUP ####################
    ################################################
    # Used for testing mostly
    def _setup(self):
        print("Initializing rhetorical clusterer")
        self.RhetoricalClusterer.initialize_clusterer()
        print("clustering rhetorically")
        self.RhetoricalClusterer.cluster_sequences()
        print("Getting gesture features")
        self.GestureClusterer.df = self.GestureClusterer.assign_feature_vectors()

    # splice gestures when there seems to be no movement or speaking
    def splice_gestures(self):
        new_df_maybe = self.GestureSplicer.splice_gestures(self.df)
        return new_df_maybe

    def initialize_rhetorical_clusterer(self):
        self.RhetoricalClusterer.initialize_clusterer(self.df)

    def cluster_rhetorical(self):
        rhetorical_clusters = self.RhetoricalClusterer.cluster_sequences()
        return rhetorical_clusters

    def load_gestures(self):
        ## for testing, so it doesn't take so long to get the file.
        if self.speaker == "test":
            fp = os.path.join(os.getcwd(), "test_agd.json")    # hacky
            if not os.path.exists(fp):
                fp = os.path.join(os.getcwd(), "data_analysis", "test_agd.json")  # ha
            d = get_data_from_path(fp)
            return d
        else:
            agd_bucket = "all_gesture_data"
            try:
                print("trying to get data from cloud from %s, %s" % (agd_bucket, "%s_agd.json" % self.speaker))
                d = get_data_from_blob(agd_bucket, "%s_agd.json" % self.speaker)
                return d
            except:
                print("No speaker gesture data found in %s for speaker %s" % (agd_bucket, self.speaker))
                print("Try running data_management_scripts/get_keyframes_for_gestures")

    def get_transcript(self):
        fp = "temp.json"
        if self.gesture_transcript:
            return
        elif self.speaker == "test":
            fp = os.path.join(os.getcwd(), "test_timings_with_transcript.json")    # hacky
            if not os.path.exists(fp):
                fp = os.path.join(os.getcwd(), "data_analysis", "test_timings_with_transcript.json")  # hacky
        else:       # TODO make this smarter so if it's already downloaded it won't download again
            download_blob(self.full_transcript_bucket,
                          "%s_timings_with_transcript.json" % self.speaker,
                          fp)

        gesture_transcript = read_data(fp)
        return gesture_transcript

    def get_df(self):
        # You can delete this at any time to re-get the test data the old, slow way.
        # For example, to test a  new splicing mechanism.
        if self.speaker == 'test':
            print("Loading test pickle")
            df = pd.read_pickle('test_rhetorical_5_13.pkl')
            return df.reset_index(drop=True)

        print("loading gestures")
        motion_data = self.load_gestures()

        print("getting transcript")
        transcripts = self.get_transcript()['phrases']

        if not motion_data or not transcripts:
            print("could not get motion or transcript for ", self.speaker)

        ids = [g['id'] for g in motion_data]

        print("converting data")
        data = {}
        for i in ids:
            motion = [el['keyframes'] for el in motion_data if el['id'] == i]
            text = [el for el in transcripts if el['id'] == i]
            if not motion:
                print("no motion data found for gesture ", i)
            if not text:
                print("no text data found for gesture ", i)
            if not motion or not text:
                continue
            text = text[0]
            motion = motion[0]

            data[i] = {
                'id': text['id'],
                'speaker': text['speaker'],
                'video_fn': text['phase']['video_fn'],
                'transcript':  text['phase']['transcript'],
                'start_seconds':  text['phase']['start_seconds'],
                'end_seconds': text['phase']['end_seconds'],
                'words': text['words'],
                'keyframes': motion
            }

        return pd.DataFrame.from_dict(data).T.reset_index()

    def get_motion_features(self):
        print("adding motion feature vector to gestures")
        try:
            motion_feature_vecs = self.GestureClusterer.df['motion_feature_vec']
            return motion_feature_vecs
        except KeyError:
            print("No motion feature vectors found. Try running GestureClusterer._assign_feature_vectors()")

    def get_index_by_gesture_id(self, gid):
        ids = self.df.index[self.df['id'] == gid].tolist()
        if ids:
            return ids[0]
        return None

    # TODO df-ify this
    # def downsample_speaker(self, speaker="angelica", n=1000):
    #    print("sampling out angelica speakers")
    #    speaker_sentences = [g for g in self.agd if self.get_gesture_by_id(g['id'])['speaker'] == speaker]
    #    print("getting all non-angelica speakers")
    #    new_agd = [g for g in self.agd if g not in speaker_sentences]
    #    print("sampling and recombining")
    #    angelica_sample = random.sample(speaker_sentences, n)
    #    agd = new_agd + angelica_sample
    #    self.agd = agd

    def cluster_gestures(self, max_number_clusters=None, max_cluster_distance=None, gesture_features=GESTURE_FEATURES):
        self.GestureClusterer = self.GestureClusterer(self.df)
        self.GestureClusterer.cluster_gestures(max_number_clusters=max_number_clusters,
                                               max_cluster_distance=max_cluster_distance,
                                               gesture_features=gesture_features)

    def get_gesture_ids_fewer_than_n_words(self, n):
        fewer_than_n = self.df[self.df['words'].apply(lambda x: len(x) <= n)]
        return fewer_than_n['id'].tolist()

    # can be used to replace GSM.agd
    def get_gestures_motion_under_time(self, time):
        ids = [g['id'] for g in self.gesture_transcript['phrases'] if (g['phase']['end_seconds'] - g['phase']['start_seconds']) < time]
        gests = [self.get_gesture_motion_by_id(i) for i in ids]
        return gests

    # can be used to replace GSM.gesture_transcript
    def get_gestures_transcript_under_time(self, time):
        ids = [g['id'] for g in self.gesture_transcript['phrases'] if (g['phase']['end_seconds'] - g['phase']['start_seconds']) < time]
        gests = [self.get_gesture_by_id(i) for i in ids]
        return gests

    def get_avg_frames_for_gesture(self):
        return np.array([len(k) for k in self.df['keyframes']]).mean()

    def get_hist_of_lengths(self, df=None):
        if df is not None:
            df = self.df
        lengths = [len(k) for k in df['keyframes']]
        plt.hist(lengths, bins=[5, 10, 20, 50, 75, 100, 120, 150, 180, 200, 250, 300, 350, 400, 450, 500, 550])


    ###########################################
    ################ REPORTING ################
    ###########################################
    def report_gesture_clusters(self):
        self.GestureClusterer.report_clusters()

    ###################################################
    ################ DATA MANIPULATION ################
    ###################################################
    # TODO df-ify this
    def filter_agd(self, exclude_ids):
        return self.df[~self.df['id'].isin(exclude_ids)]

    def get_gesture_by_id(self, g_id):
        i = self.get_index_by_gesture_id(g_id)
        return self.df.iloc[i]

    def get_gesture_motion_by_id(self, g_id):
        i = self.get_index_by_gesture_id(g_id)
        dat = self.df.iloc[i]['keyframes']
        return dat

    def get_gesture_transcript_by_id(self, g_id):
        i = self.get_index_by_gesture_id(g_id)
        dat = self.df.iloc[i]['transcript']
        return dat

    ##############################################
    ################ VIDEO THINGS ################
    ##############################################
    def get_gesture_video_clip_by_gesture_id(self, g_id, folder=""):
        g = self.get_gesture_by_id(g_id)
        self.VideoManager.get_video_clip(g['video_fn'], g['start_seconds'], g['end_seconds'], folder=folder)

    def get_gesture_audio_properties_by_gesture_id(self, g_id):
        g = self.get_gesture_by_id(g_id)
        p = g['phase']
        self.VideoManager.get_audio_features(p['video_fn'], p['start_seconds'], p['end_seconds'])

    ##############################################
    ################ DATA PEEPIN' ################
    ##############################################
    ## search for a specific phrase that may appear in the transcript of these gestures.
    # def get_gesture_clusters_by_transcript_phrase(self, phrase):

    # takes sentence cluster ID
    # returns list of all gesture clusters in which corresponding
    # sentences appear
    # def get_gesture_clusters_for_sentence_cluster(self, s_cluster_id):

    # have gesture_cluster_id in them, match those?
    def get_gesture_cluster_id_for_gesture(self, g_id):
        for k in self.GestureClusterer.clusters:
            g_ids = [g['id'] for g in self.GestureClusterer.clusters[k]['gesture_ids']]
            if g_id in g_ids:
                return k

    ################################################
    ################ VISUALIZATIONS ################
    ################################################
    def get_gesture_cluster_wordcount_data(self):
        wc = []
        for k in self.GestureClusterer.clusters:
            c = self.GestureClusterer.clusters[k]
            for g_id in c['gesture_ids']:
                s = self.get_gesture_transcript_by_id(g_id)
                wc.append(len(nltk.word_tokenize(s)))
        return wc

    def histogram_of_word_count(self):
        wc = self.get_gesture_cluster_wordcount_data()
        n, bins, patches = plt.hist(wc, bins=30, range=[0,100], normed=True, facecolor='green', alpha=0.75)
        plt.xlabel('wordcount')
        plt.ylabel('num occurances')
        # plt.axes([0, 200, 0, .03])
        plt.grid(True)
        plt.show()

    def histogram_of_gesture_lengths(self):
        time_lengths = [(g['phase']['end_seconds'] - g['phase']['start_seconds']) for g in self.gesture_transcript['phrases']]
        plt.hist(time_lengths, color='blue', edgecolor='black',
                bins=int(180 / 5))

    def get_closest_gestures_in_gesture_cluster(self, cluster_id):
        c = self.GestureClusterer.clusters[cluster_id]
        (g1, g2) = (0, 0)
        min_d = 1000
        print("exploring distances for %s gestures" % str(len(c['gestures'])))
        for i in tqdm(list(range(0, len(c['gestures'])))):
            g = c['gestures'][i]
            for j in range(0, len(c['gestures'])):
                comp = c['gestures'][j]
                dist = np.linalg.norm(np.array(g['feature_vec']) - np.array(comp['feature_vec']))
                # don't want to be comparing same ones.
                if dist < min_d and i != j:
                    min_d = dist
                    (g1, g2) = (g['id'], comp['id'])
        return (g1, g2)

    def get_furthest_gestures_in_gesture_cluster(self, cluster_id):
        c = self.GestureClusterer.clusters[cluster_id]
        (g1, g2) = (0, 0)
        max_d = 0
        print("exploring distances for %s gestures" % str(len(c['gestures'])))
        for i in tqdm(list(range(0, len(c['gestures'])))):
            g = c['gestures'][i]
            for j in range(0, len(c['gestures'])):
                comp = c['gestures'][j]
                dist = np.linalg.norm(np.array(g['feature_vec']) - np.array(comp['feature_vec']))
                # don't want to be comparing same ones.
                if dist > max_d:
                    max_d = dist
                    (g1, g2) = (g['id'], comp['id'])
        return g1, g2

    def get_speakers_by_gesture_cluster(self, g_cluster_id):
        c = self.GestureClusterer.clusters[g_cluster_id]
        speakers = [self.get_gesture_by_id(g['id'])['speaker'] for g in c['gestures']]
        counts = [speakers.count(s) for s in speakers]
        both = sorted(list(set(zip(speakers,counts))), key=lambda x: x[1], reverse=True)
        return both

    ####################################################################
    ####################### WORD CLOUD STUFF ###########################
    ####################################################################
    def filter_syntax(self, words, filter_in="", filter_out=""):
        w = words
        if filter_in:
            w = filter_words_by_syntax(w, filter_in)
        if filter_out:
            w = filter_words_out_by_syntax(w, filter_out)
        return w

    def get_words_by_gesture_cluster(self, g_cluster_id):
        c = self.GestureClusterer.clusters[g_cluster_id]
        transcripts = self.df.loc[self.df['id'].isin(c['gesture_ids'])]['transcript'].tolist()
        return " ".join(transcripts).split(" ")

    def create_word_cloud_by_gesture_cluster(self, g_cluster_id, filter_in_syntax="", filter_out_syntax="", stopwords=[]):
        stopwords = set(STOPWORDS)
        # stopwords.update(["music", "kind", "really", "thing", "know", 'people', 'one'])
        all_words = self.get_words_by_gesture_cluster(g_cluster_id)
        self.filter_syntax(all_words, filter_in_syntax, filter_out_syntax)
        all_words = " ".join(all_words)
        wordcloud = WordCloud(background_color="white").generate(all_words)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")

    def show_wordclouds_by_gesture_clusters(self, g_cluster_ids=None, filter_in_syntax="", filter_out_syntax=""):
        g_cluster_ids = g_cluster_ids if g_cluster_ids else [y[0] for y in sorted([(c, len(self.GestureClusterer.clusters[c]['gestures'])) for c in list(self.GestureClusterer.clusters.keys())], key=lambda x: x[1])[-9:]]
        for i in range(0, len(g_cluster_ids)):
            print(g_cluster_ids[i])
            plt.subplot(3, 3, i+1)
            self.create_word_cloud_by_gesture_cluster(g_cluster_ids[i], filter_in_syntax, filter_out_syntax)
            plt.text(0.5, 0.5, str(g_cluster_ids[i]), fontsize=12)
        plt.show()

    ############################################################
    ##################### Data Set Stuff #######################
    ############################################################
    def show_pie_of_speakers(self):
        self.df.speaker.value_counts().plot(kind='pie')

    #####################################################
    ############### Machine Learning Stats ##############
    #####################################################
    def get_silhouette_scores_for_all_gesture_clusters(self):
        scores = []
        for k in self.GestureClusterer.clusters:
            scores.append(self.GestureClusterer.get_silhouette_score(k))
        s = np.array(scores)
        print("number of clusters: %s" % len(self.GestureClusterer.clusters))
        print("avg silhouette: %s" % np.average(s))
        print("min silhouette: %s" % np.min(s))
        print("max silhouette: %s" % np.max(s))
        print("sd: %s" % np.std(s))
        return s

    def get_silhouette_rhetorical_vs_gesture(self):
        rhetoric_clusters = self.RhetoricalClusterer.cluster_sequences()
        silhouettes = []
        for k in rhetoric_clusters.keys():
            silhouettes.append(self.GestureClusterer.get_silhouette_score_for_alternative_clustering(rhetoric_clusters, k))
        silhouettes = np.array(silhouettes)
        return silhouettes.mean(), silhouettes.std()
