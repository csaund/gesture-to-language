#!/usr/bin/env pythons
from GestureClusterer import *
from SentenceClusterer import *
import json
import os
from termcolor import colored
import numpy as np

devKey = str(open("%s/devKey" % os.getenv("HOME"), "r").read()).strip()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "%s/google-creds.json" % os.getenv("HOME")

from google.cloud import storage
from common_helpers import *

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
# GSM = GestureSentenceManager("conglomerate")
# GSM.load_gestures()  TODO maybe put this in the initialization?
# GSM.cluster_gestures()    or    GSM.cluster_gestures_under_n_words(10)
# report = GSM.report_clusters()
# GSM.print_sentences_by_cluster(0)
# GSM.cluster_sentences_gesture_independent()    or     GSM.cluster_sentences_gesture_independent_under_n_words(10)

#

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

    def _initialize_sentence_clusterer(self):
        self.SentenceClusterer = SentenceClusterer(self.speaker)
        # now we have clusters, now need to get the corresponding sentences for those clusters.
    def cluster_sentences_gesture_independent(self):
        self.SentenceClusterer.cluster_sentences()
        self.sentenceClusters = self.SentenceClusterer.clusters

    def cluster_sentences_gesture_independent_under_n_words(self, n):
        ids_fewer_than_n = self.get_gesture_ids_fewer_than_n_words(n)
        exclude_ids = [g['id'] for g in self.gesture_transcript['phrases'] if g['id'] not in ids_fewer_than_n]
        self.SentenceClusterer.cluster_sentences(gesture_data=None, min_cluster_sim=0.5, max_cluster_size=90, max_number_clusters=1000, exclude_gesture_ids=exclude_ids)
        self.sentenceClusters = self.SentenceClusterer.clusters

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
        for k in self.sentenceClusters.keys():
            c = self.sentenceClusters[k]
            ids = [g['id'] for g in c['gestures']]
            s_ids.append(ids)
        s_ids = flatten(s_ids)

        g_ids = []
        for k in self.gestureClusters.keys():
            c = self.gestureClusters[k]
            ids = [g['id'] for g in c['gestures']]
            g_ids.append(ids)
        g_ids = flatten(g_ids)

        diff = list(set(s_ids).symmetric_difference(set(g_ids)))


    def cluster_gestures_under_n_words(self, n):
        ids_fewer_than_n = self.get_gesture_ids_fewer_than_n_words(n)
        exclude_ids = [g['id'] for g in self.gesture_transcript['phrases'] if g['id'] not in ids_fewer_than_n]
        self.cluster_gestures(exclude_ids)

    def cluster_gestures(self, exclude_ids=[]):
        if len(exclude_ids):
            self.GestureClusterer = GestureClusterer(self.filter_agd(exclude_ids))
        else:
            self.GestureClusterer = GestureClusterer(self.agd)
        self.GestureClusterer.cluster_gestures()
        self.gestureClusters = self.GestureClusterer.clusters

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
        c = self.gestureClusters[g_cluster_id]
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
            c_ids.append(self.get_cluster_id_for_gesture(g))
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
            g_cluster_ids = get_gesture_clusters_for_sentence_cluster(k)
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
                unique_matches.append((g_cluster_id, s_cluster_ids))
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
                unique_matches.append((k, g_cluster_ids))
        print "number of gestures per sentence cluster: %s" % str(lens)
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
        key = self.sentenceClusters.keys()[0]
        if 'gesture_cluster_ids' not in self.sentenceClusters[key].keys():
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
        key = self.SentenceClusterer.clusters.keys()[0]
        if not self.SentenceClusterer.clusters[key]['gesture_cluster_ids']:
            self.assign_gesture_cluster_ids_for_sentence_clusters()
        columns = ["SCID", "GCID"]
        rows = []
        for k in self.SentenceClusterer.clusters:
            c = self.SentenceClusterer.clusters[k]
            for g in c['gesture_cluster_ids']:
                rows.append({'SCID': float(k), 'GCID': float(g) * 10, 'SID': float(k)})
        df = pd.DataFrame(rows)
        return df


    def bar_chart(self):
        sentence_clusters = []
        sentence_nums = []
        num_g_clusters = []
        for k in self.sentenceClusters:
            c = self.sentenceClusters[k]
            sentence_clusters.append(k)
            sentence_nums.append(len(c['sentences']))
            num_g_clusters.append(len(c['gesture_cluster_ids']))
        columns = ["cluster", 'num_sentences', 'num_g_clusters']
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
        for k in self.gestureClusters:
            c = self.gestureClusters[k]
            for g in c['gestures']:
                s = self.get_gesture_by_id(g['id'])['phase']['transcript']
                wc.append(len(nltk.word_tokenize(s)))
        return wc

    # MAJOR TODO::: JOIN DFS ON GESTURE ID.
    def get_sentence_cluster_wordcount_data(self):
        wc = []
        for k in self.sentenceClusters:
            c = self.sentenceClusters[k]
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

    def get_words_by_sentence_cluster(self, s_cluster_id):
        c = self.sentenceClusters[s_cluster_id]
        all_words = " ".join(c['sentences'])
        return all_words.split(" ")

    def get_words_by_gesture_cluster(self, g_cluster_id):
        words = []
        c = self.gestureClusters[g_cluster_id]
        for gesture in c['gestures']:
            g = self.get_gesture_by_id(gesture['id'])
            if(g['phase']['transcript']):
                words.append(g['phase']['transcript'])
        return " ".join(words).split(" ")


    def create_word_cloud_by_gesture_cluster(self, g_cluster_id, filter_in_syntax="", filter_out_syntax="", stopwords=[]):
        # stopwords = set(STOPWORDS)
        # stopwords.update(["music", "kind", "really", "thing", "know", 'people', 'one'])
        all_words = self.get_words_by_gesture_cluster(g_cluster_id)
        if filter_in_syntax:
            all_words = filter_words_by_syntax(all_words, filter_in_syntax)
        if filter_out_syntax:
            all_words = filter_words_out_by_syntax(all_words, filter_out_syntax)
        all_words = " ".join(all_words)
        wordcloud = WordCloud(background_color="white").generate(all_words)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")


    def create_word_cloud_by_sentence_cluster(self, s_cluster_id, filter_in_syntax="", filter_out_syntax="", stopwords=[]):
        # stopwords = set(STOPWORDS)
        # stopwords.update(["music", "kind", "really", "thing", "know", 'people', 'one'])
        all_words = self.get_words_by_sentence_cluster(s_cluster_id)
        if filter_in_syntax:
            all_words = filter_words_by_syntax(all_words, filter_in_syntax)
        if filter_out_syntax:
            all_words = filter_words_out_by_syntax(all_words, filter_out_syntax)
        all_words = " ".join(all_words)
        wordcloud = WordCloud(background_color="white").generate(all_words)
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")

    ## I'm more disappointed in myself than you will ever be in me.
    def show_wordclouds_by_sentence_clusters(self, s_cluster_ids=None, filter_in_syntax="", filter_out_syntax=""):
        s_cluster_ids = s_cluster_ids if s_cluster_ids else [y[0] for y in sorted([(c, len(self.sentenceClusters[c]['sentences'])) for c in self.sentenceClusters.keys()], key=lambda x: x[1])[-9:]]
        for i in range(0, len(s_cluster_ids)):
            plt.subplot(3, 3, i+1)
            self.create_word_cloud_by_sentence_cluster(s_cluster_ids[i], filter_in_syntax, filter_out_syntax)
            plt.text(0.5, 0.5, str(s_cluster_ids[i]), fontsize=12)
        plt.show()


    ## look, no one's happy about this.
    def show_wordclouds_by_gesture_clusters(self, g_cluster_ids=None, filter_in_syntax="", filter_out_syntax=""):
        g_cluster_ids = g_cluster_ids if g_cluster_ids else [y[0] for y in sorted([(c, len(self.gestureClusters[c]['gestures'])) for c in self.gestureClusters.keys()], key=lambda x: x[1])[-9:]]
        for i in range(0, len(g_cluster_ids)):
            plt.subplot(3, 3, i+1)
            self.create_word_cloud_by_gesture_cluster(g_cluster_ids[i], filter_in_syntax, filter_out_syntax)
            plt.text(0.5, 0.5, str(g_cluster_ids[i]), fontsize=12)
        plt.show()

    # TODO chord diagram


def init_new_gsm(oldGSM):
    newGSM = GestureSentenceManager(oldGSM.speaker)
    newGSM.speaker = oldGSM.speaker
    newGSM.agd = oldGSM.agd
    newGSM.gestureClusters = oldGSM.gestureClusters
    newGSM.sentenceClusters = oldGSM.sentenceClusters
    newGSM.GestureClusterer = oldGSM.GestureClusterer
    newGSM.SentenceClusterer = oldGSM.SentenceClusterer
    return newGSM




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
