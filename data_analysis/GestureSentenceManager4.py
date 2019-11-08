#!/usr/bin/env pythons
from GestureClusterer import *
from SentenceClusterer10 import *
from VideoManager import *
import json
import os
from termcolor import colored
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from prettytable import PrettyTable

devKey = str(open("%s/devKey" % os.getenv("HOME"), "r").read()).strip()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "%s/google-creds.json" % os.getenv("HOME")

from google.cloud import storage
from common_helpers import *

from textable import TexTable

# from matplotlib_venn import venn3, venn3_circles
# from matplotlib import pyplot as plt
from pandas.plotting import parallel_coordinates
import networkx as nx
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

VERBS = ["V", "VB", "VBD", "VBD", "VBZ", "VBP", "VBN"]
NOUNS = ["NN", "NNP", "NNS"]
ADJ = ["JJ"]


## the following commands assume you have a full transcript in the cloud
## and also all the timings.
# from GestureSentenceManager import *
# GSM = GestureSentenceManager("conglomerate_under_10")
# GSM.cluster_gestures()    or    GSM.cluster_gestures_under_n_words(10)
# GSM.test_k_means_gesture_clusters()
# report = GSM.report_clusters()
# GSM.print_sentences_by_cluster(0)
# GSM.cluster_sentences_gesture_independent()    or     GSM.cluster_sentences_gesture_independent_under_n_words(10)
# GSM.assign_gesture_cluster_ids_for_sentence_clusters()

#
#
# from GestureSentenceManager import *
# GSM = GestureSentenceManager("conglomerate_under_10")
# GSM.GestureClusterer.cluster_gestures_disparate_seeds(None, max_cluster_distance=0.03, max_number_clusters=27)
# GSM.cluster_sentences_gesture_independent()

## manages gesture and sentence stuff.
class GestureSentenceManager():
    def __init__(self, speaker, seeds=[]):
        ## this is where the magic is gonna happen.
        ## get all the gestures
        self.speaker = speaker
        self.cluster_bucket_name = "%s_clusters" % speaker
        self.full_transcript_bucket = "full_timings_with_transcript_bucket"
        self.gesture_transcript = None
        self.gesture_sentence_clusters = {}
        self.get_transcript()
        self.agd = None
        self._initialize_sentence_clusterer()
        self.VideoManager = VideoManager()
        print "loading gestures"
        self.load_gestures()
        self.GestureClusterer = GestureClusterer(self.agd)

    def _initialize_sentence_clusterer(self):
        self.SentenceClusterer = SentenceClusterer(self.speaker)
        # now we have clusters, now need to get the corresponding sentences for those clusters.
    def cluster_sentences_gesture_independent(self):
        self.agd = [g for g in self.agd if g['id'] not in self.GestureClusterer.drop_ids]
        ids_for_sentences = [g['id'] for g in self.agd]
        self.SentenceClusterer.cluster_sentences(exclude_gesture_ids=self.GestureClusterer.drop_ids, include_ids=ids_for_sentences)

    def cluster_sentences_gesture_independent_under_n_words(self, n):
        ids_fewer_than_n = self.get_gesture_ids_fewer_than_n_words(n)
        exclude_ids = [g['id'] for g in self.gesture_transcript['phrases'] if g['id'] not in ids_fewer_than_n]
        exclude_ids = exclude_ids + self.GestureClusterer.drop_ids
        self.SentenceClusterer.cluster_sentences(gesture_data=None, min_cluster_sim=0.5, max_cluster_size=90, max_number_clusters=1000, exclude_gesture_ids=exclude_ids)

    def report_gesture_clusters(self):
        self.GestureClusterer.report_clusters()

    def report_sentence_clusters(self):
        self.SentenceClusterer.report_clusters()


    def load_gestures(self):
        self.agd = {}
        agd_bucket = "all_gesture_data"
        try:
            print "trying to get data from cloud from %s, %s" % (agd_bucket, "%s_agd.json" % self.speaker)
            d = get_data_from_blob(agd_bucket, "%s_agd.json" % self.speaker)
            self.agd = d
        except:
            print "No speaker gesture data found in %s for speaker %s" % (agd_bucket, self.speaker)
            print "Try running data_management_scripts/get_keyframes_for_gestures"

    def compare_ids_in_sentence_gesture_clusters(self):
        s_ids = []
        for k in self.SentenceClusterer.clusters.keys():
            c = self.SentenceClusterer.clusters[k]
            ids = [g['id'] for g in c['gestures']]
            s_ids.append(ids)
        s_ids = flatten(s_ids)

        g_ids = []
        for k in self.GestureClusterer.clusters.keys():
            c = self.GestureClusterer.clusters[k]
            ids = [g['id'] for g in c['gestures']]
            g_ids.append(ids)
        g_ids = flatten(g_ids)

        diff = list(set(s_ids).symmetric_difference(set(g_ids)))


    def cluster_gestures_under_n_words(self, n, max_number_clusters=0):
        ids_fewer_than_n = self.get_gesture_ids_fewer_than_n_words(n)
        exclude_ids = [g['id'] for g in self.gesture_transcript['phrases'] if g['id'] not in ids_fewer_than_n]
        self.cluster_gestures(exclude_ids, max_number_clusters)

    def cluster_gestures(self, exclude_ids=[], max_number_clusters=0):
        if len(exclude_ids):
            self.GestureClusterer = GestureClusterer(self.filter_agd(exclude_ids))
        else:
            self.GestureClusterer = GestureClusterer(self.agd)
        self.GestureClusterer.cluster_gestures(None, 0.03, max_number_clusters)

    def filter_agd(self, exclude_ids):
        agd = [d for d in self.agd if d['id'] not in exclude_ids]
        return agd

    def print_sentences_by_cluster(self, cluster_id):
        sents = self.get_sentences_by_cluster(cluster_id)
        empties = 0
        c = 0
        colors = ['red', 'blue']
        for i, s in enumerate(sents):
            if s:
                print colored("%s. %s" % (i, s), colors[c])
                if c:
                    c = 0
                else:
                    c = 1
            else:
                empties += 1
        print "Along with %s empty strings." % empties
        print

    def get_sentences_by_cluster(self, cluster_id):
        self.get_transcript()
        gesture_ids = self.GestureClusterer.get_gesture_ids_by_cluster(cluster_id)
        p = self.gesture_transcript['phrases']
        sentences = [d['phase']['transcript'] for d in p if d['id'] in gesture_ids]
        return sentences


    ## have to do parallel thing before this
    def get_sentence_cluster_ids_by_gesture_cluster_id(self, g_cluster_id):
        matches = [self.SentenceClusterer.clusters[k]['cluster_id'] for k in self.SentenceClusterer.clusters.keys() if g_cluster_id in self.SentenceClusterer.clusters[k]['gesture_cluster_ids']]
        return matches


    def get_sentence_clusters_by_gesture_clusters(self):
        if(self.gesture_sentence_clusters):
            return self.gesture_sentence_clusters
        for k in self.GestureClusterer.clusters:
            g_ids = [g['id'] for g in self.GestureClusterer.clusters[k]['gestures']]
            gests = self.get_gestures_by_ids(g_ids)
            self.gesture_sentence_clusters[k] = self.SentenceClusterer.create_new_cluster_by_gestures(gests)
        return self.gesture_sentence_clusters
        # well now we have the sentence clusters for the gestures... can compare some stuff?

    def get_transcript(self):
        if self.gesture_transcript:
            return
        fp = "temp.json"
        download_blob(self.full_transcript_bucket,
                      "%s_timings_with_transcript.json" % self.speaker,
                      fp)
        self.gesture_transcript = read_data(fp)
        os.remove(fp)

    def get_gesture_by_id(self, g_id):
        # because this returns list of matching items, and only one item will match,
        # we just take the first element and use that.
        g_trans = self.get_gesture_transcript_by_id(g_id)
        g_motion = self.get_gesture_motion_by_id(g_id)
        g_trans['keyframes'] = g_motion['keyframes']
        return g_trans

    def get_gesture_vid_time_by_id(self, g_id):
        g = self.get_gesture_by_id(g_id)
        vid = g['phase']['video_fn']
        start = g['phase']['start_seconds']
        end = g['phase']['end_seconds']

    def get_gesture_motion_by_id(self, g_id):
        dat = [d for d in self.agd if d['id'] == g_id]
        return dat[0]

    def get_gesture_transcript_by_id(self, g_id):
        self.get_transcript()
        p = self.gesture_transcript['phrases']
        dat = [d for d in p if d['id'] == g_id]
        # because this returns list of matching items, and only one item will match,
        # we just take the first element and use that.
        return dat[0]

    def get_gestures_by_ids(self, g_ids):
        self.get_transcript()
        p = self.gesture_transcript['phrases']
        dat = [d for d in p if d['id'] in g_ids]
        return dat

    def upload_clusters(self):
        self.GestureClusterer.write_clusters()
        upload_blob(self.cluster_bucket_name, self.GestureClusterer.cluster_file, self.cluster_bucket_name)


    ## search for a specific phrase that may appear in the transcript of these gestures.
    def get_gesture_clusters_by_transcript_phrase(self, phrase):
        clusters_containing_phrase = []
        for k in self.GestureClusterer.clusters:
            g_ids = self.GestureClusterer.get_gesture_ids_by_cluster(k)
            gests = self.get_gestures_by_ids(g_ids)
            transcripts = [g['phase']['transcript'] for g in gests]
            count = 0
            for t in transcripts:
                if phrase in t:
                    count += 1
            if count:
                clusters_containing_phrase.append((k, count))
        return sorted(clusters_containing_phrase, key=lambda x: x[1], reverse=True)


    def create_word_cloud_by_gesture_cluster(self, g_cluster_id):
        words = []
        c = self.GestureClusterer.clusters[g_cluster_id]
        for gesture in c['gestures']:
            g = self.get_gesture_by_id(gesture['id'])
            words.append(g['phase']['transcript'])
        words


    # takes sentence cluster ID
    # returns list of all gesture clusters in which corresponding
    # sentences appear
    def get_gesture_clusters_for_sentence_cluster(self, s_cluster_id):
        c = self.SentenceClusterer.clusters[s_cluster_id]
        g_cluster_ids = [g['id'] for g in c['gestures']]
        c_ids = []
        for g in g_cluster_ids:
            c_ids.append(self.get_gesture_cluster_id_for_gesture(g))
        return list(set(c_ids))
        # now get the cluster id for each gesture


    # TODO I can make this more clever -- gesture cluster gestures
    # have gesture_cluster_id in them, match those?
    def get_gesture_cluster_id_for_gesture(self, g_id):
        for k in self.GestureClusterer.clusters:
            g_ids = [g['id'] for g in self.GestureClusterer.clusters[k]['gestures']]
            if g_id in g_ids:
                return k



    def assign_gesture_cluster_ids_for_sentence_clusters(self):
        for k in self.SentenceClusterer.clusters:
            g_cluster_ids = self.get_gesture_clusters_for_sentence_cluster(k)
            self.SentenceClusterer.clusters[k]['gesture_cluster_ids'] = g_cluster_ids


    ## for each gesture cluster, how many sentence clusters are represented?
    def get_sentence_cluster_ids_for_gesture_cluster(self, g_cluster_id):
        c_ids = []
        for k in self.SentenceClusterer.clusters:
            if g_cluster_id in self.SentenceClusterer.clusters[k]['gesture_cluster_ids']:
                c_ids.append(k)
        return c_ids

    def report_gesture_cluster_overlap_with_sentence_clusters(self):
        lens = []
        unique_matches = []
        for g_cluster_id in self.GestureClusterer.clusters:
            s_cluster_ids = self.get_sentence_cluster_ids_for_gesture_cluster(g_cluster_id)
            print "Sentence clusters represented in g_cluster %s: %s" % (g_cluster_id, s_cluster_ids)
            lens.append(len(s_cluster_ids))
            if len(s_cluster_ids) == 1:
                unique_matches.append((g_cluster_id, s_cluster_ids[0]))
        print "avg number sentence clusters: %s" % str(float(sum(lens)) / float(len(lens)))
        print "sd of lengths of gesture clusters: %s" % str(np.std(lens))
        print
        print "number of unique gesture-sentence matches: %s/%s" % (len(unique_matches), len(self.GestureClusterer.clusters))
        print "unique matches: %s" % unique_matches


    def report_sentence_cluster_overlap_with_gesture_clusters(self):
        lens = []
        unique_matches = []
        for k in self.SentenceClusterer.clusters:
            g_cluster_ids = self.SentenceClusterer.clusters[k]['gesture_cluster_ids']
            lens.append(len(g_cluster_ids))
            if len(g_cluster_ids) == 1:
                unique_matches.append((k, g_cluster_ids[0]))
        print "number of gesture clusters per sentence cluster: %s" % str(lens)
        print "avg number of gesture clusters for sentence cluster: %s" % str(float(sum(lens)) / float(len(lens)))
        print "sd of lengths of gesture clusters: %s" % str(np.std(lens))
        print
        print "number of unique matches: %s/%s" % (len(unique_matches), len(self.SentenceClusterer.clusters))
        print "unique matches: %s" % unique_matches



    def plot_sentence_gesture_map_parallel(self):
        df = self.get_sentence_gesture_data_parallel()
        plt.figure()
        parallel_coordinates(df, 'SID').legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.show()

    def network(self):
        df = self.get_sentence_gesture_data_network()
        G = nx.from_pandas_edgelist(df, 'from', 'to')
        nx.draw(G, with_labels=True)
        plt.show()

    def get_sentence_gesture_data_network(self):
        key = self.SentenceClusterer.clusters.keys()[0]
        if 'gesture_cluster_ids' not in self.SentenceClusterer.clusters[key].keys():
            self.assign_gesture_cluster_ids_for_sentence_clusters()
        from_ = []
        to_ = []
        for k in self.SentenceClusterer.clusters:
            c = self.SentenceClusterer.clusters[k]
            for g in c['gesture_cluster_ids']:
                from_.append(k)
                to_.append(g)
        df = pd.DataFrame({"from": from_, "to": to_})
        return df

    def get_sentence_gesture_data_parallel(self):
        self.assign_gesture_cluster_ids_for_sentence_clusters()
        columns = ["SCID", "GCID"]
        rows = []
        for k in self.SentenceClusterer.clusters:
            c = self.SentenceClusterer.clusters[k]
            for g in c['gesture_cluster_ids']:
                rows.append({'SCID': float(k), 'GCID': float(g) * 10, 'SID': float(k)})
        df = pd.DataFrame(rows)
        return df


    def report_sentence_cluster_by_gesture_cluster(self):
        gs = self.GestureClusterer.clusters.keys()
        sents = []
        for g in self.GestureClusterer.clusters:
            s_keys = self.get_sentence_cluster_ids_by_gesture_cluster_id(g)
            sents.append(s_keys)
        t = PrettyTable()
        t.add_column("gesture cluster", gs)
        t.add_column("sentence cluster ids", sents)
        print(t)

    def report_gesture_cluster_by_sentence_cluster(self):
        self.assign_gesture_cluster_ids_for_sentence_clusters()
        ss = self.SentenceClusterer.clusters.keys()
        gs = []
        for s in self.SentenceClusterer.clusters:
            gs.append(self.SentenceClusterer.clusters[s]['gesture_cluster_ids'])
        t = PrettyTable()
        t.add_column("sentence cluster", ss)
        t.add_column("gesture cluster ids", gs)
        print(t)

    def bar_chart(self):
        sentence_clusters = []
        sentence_nums = []
        num_g_clusters = []
        for k in self.SentenceClusterer.clusters:
            c = self.SentenceClusterer.clusters[k]
            sentence_clusters.append(k)
            sentence_nums.append(len(c['sentences']))
            num_g_clusters.append(len(c['gesture_cluster_ids']))
        columns = ["sentence cluster id", 'num_sentences', 'num_g_clusters']
        df = pd.DataFrame({'cluster': sentence_clusters, 'num_sentences': sentence_nums, 'num_g_clusters': num_g_clusters})
        df.plot(x='cluster', y=["num_sentences", "num_g_clusters"], kind="bar")
        plt.show()


    def scatter(self):
        sentence_clusters = []
        sentence_nums = []
        num_g_clusters = []
        for k in self.SentenceClusterer.clusters:
            c = self.SentenceClusterer.clusters[k]
            sentence_clusters.append(k)
            sentence_nums.append(len(c['sentences']))
            num_g_clusters.append(len(c['gesture_cluster_ids']))
        columns = ["cluster", 'num_sentences', 'num_g_clusters']
        df = pd.DataFrame({'cluster': sentence_clusters, 'num_sentences': sentence_nums, 'num_g_clusters': num_g_clusters})
        df.plot.scatter(x='num_sentences', y="num_g_clusters")
        plt.show()

    def get_wordcount_data(self):
        wordcounts = [float(len(g['phase']['transcript'].split(' '))) for g in self.gesture_transcript['phrases']]
        wc = [w for w in wordcounts if w > 1]
        return wc

    # MAJOR TODO::: JOIN DFS ON GESTURE ID.
    def get_gesture_cluster_wordcount_data(self):
        wc = []
        for k in self.GestureClusterer.clusters:
            c = self.GestureClusterer.clusters[k]
            for g in c['gestures']:
                s = self.get_gesture_by_id(g['id'])['phase']['transcript']
                wc.append(len(nltk.word_tokenize(s)))
        return wc

    # MAJOR TODO::: JOIN DFS ON GESTURE ID.
    def get_sentence_cluster_wordcount_data(self):
        wc = []
        for k in self.SentenceClusterer.clusters:
            c = self.SentenceClusterer.clusters[k]
            for s in c['sentences']:
                wc.append(len(nltk.word_tokenize(s)))
        return wc

    def histogram_of_word_count(self, wc_data=None):
        wc = np.array(self.get_wordcount_data())
        if wc_data == "sentence":
            wc = self.get_sentence_cluster_wordcount_data()
        elif wc_data == "gesture":
            wc = self.get_gesture_cluster_wordcount_data()
        n, bins, patches = plt.hist(wc, bins=30, range=[0,100], normed=True, facecolor='green', alpha=0.75)
        plt.xlabel('wordcount')
        plt.ylabel('num occurances')
        # plt.axes([0, 200, 0, .03])
        plt.grid(True)
        plt.show()


    def get_gesture_ids_fewer_than_n_words(self, n):
        ids = [g['id'] for g in self.gesture_transcript['phrases'] if len(g['phase']['transcript'].split(' ')) < n and len(g['phase']['transcript'].split(' ')) > 1]
        return ids


    def get_closest_gestures_in_gesture_cluster(self, cluster_id):
        c = self.GestureClusterer.clusters[cluster_id]
        (g1, g2) = (0, 0)
        min_d = 1000
        print "exploring distances for %s gestures" % str(len(c['gestures']))
        for i in tqdm(range(0, len(c['gestures']))):
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
        print "exploring distances for %s gestures" % str(len(c['gestures']))
        for i in tqdm(range(0, len(c['gestures']))):
            g = c['gestures'][i]
            for j in range(0, len(c['gestures'])):
                comp = c['gestures'][j]
                dist = np.linalg.norm(np.array(g['feature_vec']) - np.array(comp['feature_vec']))
                # don't want to be comparing same ones.
                if dist > max_d:
                    max_d = dist
                    (g1, g2) = (g['id'], comp['id'])
        return (g1, g2)

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

    def get_words_by_sentence_cluster(self, s_cluster_id):
        c = self.SentenceClusterer.clusters[s_cluster_id]
        all_words = " ".join(c['sentences'])
        return all_words.split(" ")

    def get_words_by_gesture_cluster(self, g_cluster_id):
        words = []
        c = self.GestureClusterer.clusters[g_cluster_id]
        for gesture in c['gestures']:
            g = self.get_gesture_by_id(gesture['id'])
            if(g['phase']['transcript']):
                words.append(g['phase']['transcript'])
        return " ".join(words).split(" ")


    def create_word_cloud_by_gesture_cluster(self, g_cluster_id, filter_in_syntax="", filter_out_syntax="", stopwords=[]):
        # stopwords = set(STOPWORDS)
        # stopwords.update(["music", "kind", "really", "thing", "know", 'people', 'one'])
        all_words = self.get_words_by_gesture_cluster(g_cluster_id)
        self.filter_syntax(all_words, filter_in_syntax, filter_out_syntax)
        all_words = " ".join(all_words)
        wordcloud = WordCloud(background_color="white").generate(all_words)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")


    def create_word_cloud_by_sentence_cluster(self, s_cluster_id, filter_in_syntax="", filter_out_syntax="", stopwords=[]):
        # stopwords = set(STOPWORDS)
        # stopwords.update(["music", "kind", "really", "thing", "know", 'people', 'one'])
        all_words = self.get_words_by_sentence_cluster(s_cluster_id)
        self.filter_syntax(all_words, filter_in_syntax, filter_out_syntax)
        all_words = " ".join(all_words)
        wordcloud = WordCloud(background_color="white").generate(all_words)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")

    ## I'm more disappointed in myself than you will ever be in me.
    def show_wordclouds_by_sentence_clusters(self, s_cluster_ids=None, filter_in_syntax="", filter_out_syntax=""):
        s_cluster_ids = s_cluster_ids if s_cluster_ids else [y[0] for y in sorted([(c, len(self.SentenceClusterer.clusters[c]['sentences'])) for c in self.SentenceClusterer.clusters.keys()], key=lambda x: x[1])[-9:]]
        for i in range(0, len(s_cluster_ids)):
            print s_cluster_ids[i]
            plt.subplot(3, 3, i+1)
            self.create_word_cloud_by_sentence_cluster(s_cluster_ids[i], filter_in_syntax, filter_out_syntax)
            plt.text(0.5, 0.5, str(s_cluster_ids[i]), fontsize=12)
        plt.show()


    ## look, no one's happy about this.
    def show_wordclouds_by_gesture_clusters(self, g_cluster_ids=None, filter_in_syntax="", filter_out_syntax=""):
        g_cluster_ids = g_cluster_ids if g_cluster_ids else [y[0] for y in sorted([(c, len(self.GestureClusterer.clusters[c]['gestures'])) for c in self.GestureClusterer.clusters.keys()], key=lambda x: x[1])[-9:]]
        for i in range(0, len(g_cluster_ids)):
            print g_cluster_ids[i]
            plt.subplot(3, 3, i+1)
            self.create_word_cloud_by_gesture_cluster(g_cluster_ids[i], filter_in_syntax, filter_out_syntax)
            plt.text(0.5, 0.5, str(g_cluster_ids[i]), fontsize=12)
        plt.show()

    # TODO chord diagram


    def rank_words_by_sentence_cluster(self, cluster_id, filter_in_syntax="", filter_out_syntax=""):
        all_words = self.get_words_by_sentence_cluster(cluster_id)
        self.filter_syntax(all_words, filter_in_syntax, filter_out_syntax)
        # create a table that's nice to read.
        wds = sorted(list(set([(w, all_words.count(w)) for w in all_words])), key=lambda x: x[1], reverse=True)

        words = [w[0] for w in wds]
        counts = [w[1] for w in wds]

        t = PrettyTable()
        t.add_column("Word", words)
        t.add_column("Count", counts)
        print(t)

        df = pd.DataFrame({
            'Word': words,
            'Count': counts
        })

        df

    ############################################################
    ##################### Data Set Stuff #######################
    ############################################################
    def show_pie_of_speakers(self):
        speakers = {}
        for g in tqdm(self.agd):
            gest = self.get_gesture_by_id(g['id'])
            sp = gest['speaker']
            if sp in speakers.keys():
                speakers[sp] = speakers[sp] + 1
            else:
                speakers[sp] = 1
        labels = speakers.keys()
        sizes = speakers.values()
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True, startangle=140)
        plt.axis('equal')
        plt.savefig('speaker_distribution.png')
        plt.show()


    ############################################################
    ##################### Video Stuff ##########################
    ############################################################
    def get_gesture_video_clip_by_gesture_id(self, g_id):
        g = self.get_gesture_by_id(g_id)
        p = g['phase']
        self.VideoManager.get_video_clip(p['video_fn'], p['start_seconds'], p['end_seconds'])


    ############################################################
    ################ Gesture the plotting stuff ################
    ############################################################
    ## flip so instead of format like
    # [t1, t2, t3], [t1`, t2`, t3`], [t1``, t2``, t3``]
    # it's in the format of
    # [t1, t1`, t1``], [t2, t2`, t2``], [t3, t3`, t3``]
    def arrange_data_by_time(self, dat_vector):
        flipped_dat = []
        for i in range(len(dat_vector[0])):
            a = []
            for d in dat_vector:
                a.append(d[i])
            flipped_dat.append(a)
        return flipped_dat


    def plot_coords(self, x_y, gesture):
        coords = [d[x_y] for d in gesture['keyframes']]
        fc = self.arrange_data_by_time(coords)
        for v in fc:
            plt.plot(range(0, len(fc[0])), v)
        plt.xlabel("frame")
        plt.ylabel("%s pixel position" % x_y)
        plt.title = '%s coordinates for gesture %s' % (x_y, gesture['id'])


    def plot_both_gesture_coords(self, g_id):
        g = self.get_gesture_motion_by_id(g_id)
        plt.subplot(1,2,1)
        self.plot_coords('x', g)
        plt.subplot(1,2,2)
        self.plot_coords('y', g)
        plt.title = 'xy coordinates for gesture %s' % g['id']
        plt.savefig('%s.png' % g['id'])
        plt.show()
        return


    def plot_two_gestures(self, g_id1, g_id2):
        g1 = self.get_gesture_motion_by_id(g_id1)
        g2 = self.get_gesture_motion_by_id(g_id2)
        plt.subplot(2,2,1)
        self.plot_coords('x', g1)
        plt.subplot(2, 2, 2)
        self.plot_coords('y', g1)
        plt.subplot(2, 2, 3)
        self.plot_coords('x', g2)
        plt.subplot(2, 2, 4)
        self.plot_coords('y', g2)
        plt.title = 'xy coordinates for gesture %s and %s' % (g1['id'], g2['id'])
        plt.savefig('%s_%s.png' % (g1['id'], g2['id']))
        plt.show()
        return


    def plot_random_gestures_from_gesture_cluster(self, cluster_id):
        g1 =0
        g2 = 0
        while g1 == g2:
            g1 = self.GestureClusterer.get_random_gesture_id_from_cluster(cluster_id)
            g2 = self.GestureClusterer.get_random_gesture_id_from_cluster(cluster_id)
        print "plotting gestures for %s, %s" % (g1, g2)
        self.plot_two_gestures(g1, g2)


    def plot_closest_gestures_from_gesture_cluster(self, cluster_id):
        (g1, g2) = self.get_closest_gestures_in_gesture_cluster(cluster_id)
        print "plotting gestures for %s, %s" % (g1, g2)
        self.plot_two_gestures(g1, g2)

    def plot_furthest_gestures_from_gesture_cluster(self, cluster_id):
        (g1, g2) = self.get_furthest_gestures_in_gesture_cluster(cluster_id)
        print "plotting gestures for %s, %s" % (g1, g2)
        self.plot_two_gestures(g1, g2)

    #####################################################
    ############### Machine Learning Stats ##############
    #####################################################
    def get_silhouette_scores_for_all_gesture_clusters(self):
        scores = []
        for k in self.GestureClusterer.clusters:
            scores.append(self.GestureClusterer.get_silhouette_score(k))
        s = np.array(scores)
        print "number of clusters: %s" % len(self.GestureClusterer.clusters)
        print "avg silhouette: %s" % np.average(s)
        print "min silhouette: %s" % np.min(s)
        print "max silhouette: %s" % np.max(s)
        print "sd: %s" % np.std(s)
        return s

    def get_silhouette_scores_for_all_sentence_clusters(self):
        scores = []
        for k in self.SentenceClusterer.clusters:
            scores.append(self.SentenceClusterer.get_silhouette_score_for_cluster(k))
        s = np.array(scores)
        print "number of clusters: %s" % len(self.SentenceClusterer.clusters)
        print "avg silhouette: %s" % np.average(s)
        print "min silhouette: %s" % np.min(s)
        print "max silhouette: %s" % np.max(s)
        print "sd: %s" % np.std(s)
        return s


    def test_k_means_gesture_clusters(self, n_words=10, min_distances=[0.01, 0.03, 0.05, 0.07], ks=[10,40,60,100,150, 200, 0]):
        n_clusters = []
        max_k = []
        avgs = []
        mins = []
        maxs = []
        sd = []
        dists = []
        for k in ks:
            print "testing clustering for k=%s" % k
            for dist in min_distances:
                max_k.append(k)
                self.GestureClusterer.clear_clusters()
                self.GestureClusterer.cluster_gestures(max_cluster_distance=dist, max_number_clusters=k)
                scores = self.get_silhouette_scores_for_all_gesture_clusters()
                dists.append(dist)
                n_clusters.append(len(scores))
                avgs.append(np.average(scores))
                mins.append(np.min(scores))
                maxs.append(np.max(scores))
                sd.append(np.std(scores))

        t = PrettyTable()
        t.add_column("max k", max_k)
        t.add_column("k", n_clusters)
        t.add_column("min_dist", dists)
        t.add_column("avg silhouette", avgs)
        t.add_column("min silhouette", mins)
        t.add_column("max silhouette", maxs)
        t.add_column("sd silhouette", sd)
        print(t)



    def test_k_means_sentence_clusters(self, n_words=10, min_sims=[0.1, 0.3, 0.5, 0.7], ks=[10,40,60,100,150, 200, 0]):
        n_clusters = []
        max_k = []
        avgs = []
        mins = []
        maxs = []
        sd = []
        sims = []
        ids_for_sentences = [g['id'] for g in self.agd]
        for k in ks:
            print "testing clustering for k=%s" % k
            for sim in min_sims:
                max_k.append(k)
                self.SentenceClusterer.clear_clusters()
                self.SentenceClusterer.cluster_sentences(exclude_gesture_ids=self.GestureClusterer.drop_ids,
                                                         include_ids=ids_for_sentences,
                                                         min_cluster_sim=sim,
                                                         max_number_clusters=k)
                scores = self.get_silhouette_scores_for_all_sentence_clusters()
                sims.append(sim)
                n_clusters.append(len(scores))
                avgs.append(np.average(scores))
                mins.append(np.min(scores))
                maxs.append(np.max(scores))
                sd.append(np.std(scores))
        t = PrettyTable()
        t.add_column("max k", max_k)
        t.add_column("k", n_clusters)
        t.add_column("min_dist", dists)
        t.add_column("avg silhouette", avgs)
        t.add_column("min silhouette", mins)
        t.add_column("max silhouette", maxs)
        t.add_column("sd silhouette", sd)
        print(t)

    ## TODO ADD THIS TO GSM
    def get_sentence_stats_for_gesture_cluster(self, g_cluster_id):
        s_ids = self.get_sentence_cluster_ids_by_gesture_cluster_id(g_cluster_id)
        num_mappings = []
        other_gclusters = []
        unique_gclusts = []
        for s in s_ids:
            s_clust = self.SentenceClusterer.clusters[s]
            num_mappings.append(len(s_clust['gesture_cluster_ids'])-1)
            other_gclusters.append(s_clust['gesture_cluster_ids'])
            unique_gclusts = unique_gclusts + s_clust['gesture_cluster_ids']
        print "number of sentence clusters: %s" % str(len(s_ids)-1)
        print "min other gesture mappings: %s" % min(num_mappings)
        print "max other gesture mappings: %s" % max(num_mappings)
        print "med other gesture mappings: %s" % np.median(np.array(num_mappings))
        print "unique other gesture clusters: %s" % len(list(set(unique_gclusts)))



    def get_gesture_stats_for_sentence_cluster(self, s_cluster_id):
        g_ids = self.SentenceClusterer.clusters[s_cluster_id]['gesture_cluster_ids']
        num_mappings = []
        other_gclusters = []
        unique_sclusts = []
        for g in g_ids:
            other_g_clust = self.get_sentence_cluster_ids_by_gesture_cluster_id(g)
            num_mappings.append(len(other_g_clust)-1)
            other_gclusters.append(other_g_clust)
            unique_sclusts = unique_sclusts + other_g_clust
        print "number of sentence clusters: %s" % str(len(g_ids)-1)
        print "min other gesture mappings: %s" % min(num_mappings)
        print "max other gesture mappings: %s" % max(num_mappings)
        print "med other gesture mappings: %s" % np.median(np.array(num_mappings))
        print "unique other sentence clusters: %s" % len(list(set(unique_sclusts)))


    # for sentence cluster S and gesture cluster G, returns proportion of sentences of S which
    # are clustered in gesture cluster G
    def get_proportion_of_sentences_in_gesture_cluster(self, s_cluster_id, g_cluster_id):
        c = self.GestureClusterer.clusters[g_cluster_id]
        g_ids = [g['id'] for g in c['gestures']]
        s_cluster = self.SentenceClusterer.clusters[s_cluster_id]
        matches = [g['id'] for g in s_cluster['gestures'] if g['id'] in g_ids]
        prop = float(len(matches)) / float(len(s_cluster['gestures']))
        return prop

    # for a gesture cluster, which sentences came from which sentence clusters?
    def pie_format(self, pct, allvals):
        absolute = round(float(pct * np.sum(allvals)), 2)
        return "{:.2f}%\n({})".format(pct, absolute)

    def show_pie_sentence_clusters_for_gesture_cluster(self, g_cluster_id, exclude_sentence_clusters=[]):
        s_ids = self.get_sentence_cluster_ids_by_gesture_cluster_id(g_cluster_id)
        s_ids = [s for s in s_ids if s not in exclude_sentence_clusters]
        counts = []
        for s in s_ids:
            counts.append(self.get_proportion_of_sentences_in_gesture_cluster(s, g_cluster_id))
        # need to normalize so that sum counts >= 1
        orig = np.array(counts)
        counts = orig / orig.min()
        labels = s_ids
        fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
        wedges, texts, autotexts = ax.pie(counts, labels=labels, autopct=lambda pct: self.pie_format(pct, orig))
        plt.axis('equal')
        ax.legend(wedges, counts,
                  title="Sentence Cluster Representation",
                  loc="center left",
                  bbox_to_anchor=(1, 0, 0.5, 1))
        plt.setp(autotexts, size=8, weight="bold")
        ax.set_title("Proportional Sentence Cluster Representation in Gesture Cluster %s" % g_cluster_id)
        plt.savefig('gclust%s_sclust_distribution.png' % g_cluster_id)
        plt.show()

    def show_pie_gesture_clusters_for_sentence_cluster(self, s_cluster_id):
        sentence_cluster_gesture_ids = [g['id'] for g in self.SentenceClusterer.clusters[s_cluster_id]['gestures']]
        g_ids = []
        counts = []
        # todo shouldn't have to go through whole gesture clusters.
        for k in self.GestureClusterer.clusters:
            gesture_cluster = self.GestureClusterer.clusters[k]
            g_cluster_ids = [g['id'] for g in gesture_cluster['gestures']]
            matches = [i for i in sentence_cluster_gesture_ids if i in g_cluster_ids]
            if len(matches):
                counts.append(len(matches))
                g_ids.append(k)
        fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
        wedges, texts, autotexts = ax.pie(counts, labels=g_ids, autopct=lambda pct: self.pie_format(pct, counts))
        plt.axis('equal')
        ax.legend(wedges, counts,
                  title="Sentence Cluster Representation",
                  loc="center left",
                  bbox_to_anchor=(1, 0, 0.5, 1))
        plt.setp(autotexts, size=8, weight="bold")
        ax.set_title("Proportional Sentence Cluster Representation in Gesture Cluster %s" % s_cluster_id)
        plt.savefig('gclust%s_sclust_distribution.png' % s_cluster_id)
        plt.show()




def bl(gsm):
    for i in gsm.GestureClusterer.clusters:
        print i
        print gsm.get_sentence_stats_for_gesture_cluster(i)


def init_new_gsm(oldGSM):
    newGSM = GestureSentenceManager(oldGSM.speaker)
    newGSM.speaker = oldGSM.speaker
    newGSM.agd = oldGSM.agd
    newGSM.GestureClusterer = oldGSM.GestureClusterer
    newGSM.SentenceClusterer = oldGSM.SentenceClusterer
    return newGSM


def upload_data_under_n_words(gsm, under_words=10):
    full_timings_bucket = "full_timings_with_transcript_bucket"
    agd_bucket = "all_gesture_data"
    ids = gsm.get_gesture_ids_fewer_than_n_words(under_words)
    new_transcript = [g for g in gsm.gesture_transcript['phrases'] if len(g['phase']['transcript'].split(' ')) < under_words and len(g['phase']['transcript'].split(' ')) > 1]
    n_t = {'phrases': new_transcript}
    # upload to conglomerate_under_%s_timings_with_transcript.json to full_timings_with_transcript_bucket
    new_transcript_name = "conglomerate_under_%s_timings_with_transcript.json" % under_words
    print "uploading %s new transcript to %s / %s" % (str(len(n_t['phrases'])), full_timings_bucket, new_transcript_name)
    upload_object(full_timings_bucket, n_t, new_transcript_name)
    # upload to conglomerate_under_%s_agd.json to all_gesture_data
    new_agd = [g for g in gsm.agd if g['id'] in ids]
    new_agd_name = "conglomerate_under_%s_agd.json" % under_words
    print "uploading %s new agd to %s / %s" % (str(len(new_agd)), agd_bucket, new_agd_name)
    upload_object(agd_bucket, new_agd, new_agd_name)




def count_sentence_clusters_of_gesture(gsm, g_id):
    count = 0
    for k in gsm.sentenceClusters.keys():
        c = gsm.sentenceClusters[k]
        for g in c['gestures']:
            if g['id'] == g_id:
                count += 1
    return count


def check_sentence_cluster_counts(gsm, ids):
    counts = []
    for i in ids:
        counts.append(count_sentence_clusters_of_gesture(gsm, i))
    return counts


## STOP



## oh dang I'm being so lazy this is definitely going to bite me.
# just gonna eyeball this...
def downsample_speaker(gsm):
    print "sampling out angelica speakers"
    angelica_speaker = [g for g in gsm.agd if gsm.get_gesture_by_id(g['id'])['speaker'] == 'angelica']
    print "getting all non-angelica speakers"
    new_agd = [g for g in gsm.agd if g not in angelica_speaker]
    print "sampling and recombining"
    angelica_sample = random.sample(angelica_speaker, 1000)
    agd = new_agd + angelica_sample
    gsm.agd = agd




## INCORPORATE
def show_pie_sentence_clusters_for_gesture_cluster(gsm, g_cluster_id):
    c = gsm.GestureClusterer.clusters[g_cluster_id]
    g_ids = [g['id'] for g in c['gestures']]
    s_ids = []
    counts = []
    all_matches = []
    for s in gsm.SentenceClusterer.clusters:
        s_clust = gsm.SentenceClusterer.clusters[s]
        matches = [g['id'] for g in s_clust['gestures'] if g['id'] in g_ids]
        if len(matches):
            s_ids.append(s)
            counts.append(len(matches))
            all_matches = all_matches + matches
    unfound = [g for g in g_ids if g not in all_matches]
    labels = s_ids
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
    wedges, texts, autotexts = ax.pie(counts, labels=labels, autopct=lambda pct: func(pct, counts))
    plt.axis('equal')
    ax.legend(wedges, counts,
               title="Sentence Cluster Representation",
               loc="center left",
               bbox_to_anchor=(1, 0, 0.5, 1))
    plt.setp(autotexts, size=8, weight="bold")
    ax.set_title("Proportional Sentence Cluster Representation in Gesture Cluster %s" % g_cluster_id)
    # plt.savefig('gclust%s_sclust_distribution.png' % g_cluster_id)
    plt.show()
    print "SANITY CHECK::"
    print (sum(counts))
    print len(g_ids)
    print "unfound: %s" % unfound


def func(pct, allvals):
    absolute = int(round(float(pct*np.sum(allvals)) / 100, 2))
    return "{}".format(absolute)

def show_pie_sentence_clusters_for_gesture_cluster(gsm, g_cluster_id, exclude_sentence_clusters=[]):
    s_ids = gsm.get_sentence_cluster_ids_by_gesture_cluster_id(g_cluster_id)
    s_ids = [s for s in s_ids if s not in exclude_sentence_clusters]
    counts = []
    for s in s_ids:
        counts.append(gsm.get_proportion_of_sentences_in_gesture_cluster(s, g_cluster_id))
    # need to normalize so that sum counts >=1
    orig = np.array(counts)
    counts = orig/orig.min()
    labels = s_ids
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
    wedges, texts, autotexts = ax.pie(counts, labels=labels, autopct=lambda pct: func(pct, orig))
    plt.axis('equal')
    ax.legend(wedges, orig,
               title="Sentence Cluster Representation",
               loc="center left",
               bbox_to_anchor=(1, 0, 0.5, 1))
    plt.setp(autotexts, size=8, weight="bold")
    ax.set_title("Proportional Sentence Cluster Representation in Gesture Cluster %s" % g_cluster_id)
    # plt.savefig('gclust%s_sclust_distribution.png' % g_cluster_id)
    plt.show()



def show_pie_gesture_clusters_for_sentence_cluster(gsm, s_cluster_id):
    sentence_cluster_gesture_ids = [g['id'] for g in gsm.SentenceClusterer.clusters[s_cluster_id]['gestures']]
    g_ids = []
    counts = []
    # todo shouldn't have to go through whole gesture clusters.
    for k in gsm.GestureClusterer.clusters:
        gesture_cluster = gsm.GestureClusterer.clusters[k]
        g_cluster_ids = [g['id'] for g in gesture_cluster['gestures']]
        matches = [i for i in sentence_cluster_gesture_ids if i in g_cluster_ids]
        if len(matches):
            counts.append(len(matches))
            g_ids.append(k)
    print counts
    fig, ax = plt.subplots(figsize=(6, 3), subplot_kw=dict(aspect="equal"))
    wedges, texts, autotexts = ax.pie(counts, labels=g_ids, autopct=lambda pct: func(pct, counts))
    plt.axis('equal')
    ax.legend(wedges, counts,
              title="Sentence Cluster Representation",
              loc="center left",
              bbox_to_anchor=(1, 0, 0.5, 1))
    plt.setp(autotexts, size=8, weight="bold")
    ax.set_title("Proportional Gesture Cluster Representation in Sentence Cluster %s" % s_cluster_id)
    # plt.savefig('gclust%s_sclust_distribution.png' % s_cluster_id)
    plt.show()

## visualize gesture cluster distribution for sentence clusters