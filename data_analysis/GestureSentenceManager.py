#!/usr/bin/env pythons
from SpeakerGestureGetter import *
from GestureClusterer import *
from SentenceClusterer import *
import json
import os
from termcolor import colored

devKey = str(open("/Users/carolynsaund/devKey", "r").read()).strip()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/carolynsaund/google-creds.json"

from google.cloud import storage
from common_helpers import *

## the following commands assume you have a full transcript in the cloud
## and also all the timings.
# from GestureSentenceManager import *
# GSM = GestureSentenceManager("/Users/carolynsaund/github/gest-data/data", "rock")
# GSM.load_gestures()
# GSM.cluster_gestures()
# report = GSM.report_clusters()
# GSM.print_sentences_by_cluster(0)
# GSM.cluster_sentences_gesture_independent()

## manages gesture and sentence stuff.
class GestureSentenceManager():
    def __init__(self, base_path, speaker, seeds=[]):
        ## this is where the magic is gonna happen.
        ## get all the gestures
        self.base_path = base_path
        self.speaker = speaker
        self.SpeakerGestures = SpeakerGestureGetter(base_path, speaker)
        self.cluster_bucket_name = "%s_clusters" % speaker
        self.full_transcript_bucket = "full_timings_with_transcript_bucket"
        self.gesture_transcript = None
        self.gesture_sentence_clusters = {}
        self.get_transcript()
        self.agd = None
        self._initialize_sentence_clusterer()


    def _initialize_sentence_clusterer(self):
        self.SentenceClusterer = SentenceClusterer(self.base_path, self.speaker)
        # now we have clusters, now need to get the corresponding sentences for those clusters.
    def cluster_sentences_gesture_independent(self):
        self.SentenceClusterer.cluster_sentences()

    def report_clusters(self):
        self.GestureClusterer.report_clusters()

    def load_gestures(self):
        self.agd = self.SpeakerGestures.perform_gesture_analysis()

    def cluster_gestures(self):
        self.GestureClusterer = GestureClusterer(self.SpeakerGestures.all_gesture_data)
        self.GestureClusterer.cluster_gestures()

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
        dat = [d for d in p if d['id'] == d_id]
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
        def get_cluster_id_for_gesture(self, g_id):
            for k in self.GestureClusterer.clusters:
                g_ids = [g['id'] for g in self.GestureClusterer.clusters[k]['gestures']]
                if g_id in g_ids:
                    return k


# takes sentence cluster ID
# returns list of all gesture clusters in which corresponding
# sentences appear
def get_gesture_clusters_for_sentence_cluster(gsm, s_cluster_id):
    c = gsm.SentenceClusterer.clusters[s_cluster_id]
    g_cluster_ids = [g['id'] for g in c['gestures']]
    c_ids = []
    for g in g_cluster_ids:
        c_ids.append(get_cluster_id_for_gesture(gsm, g))
    return list(set(c_ids))
    # now get the cluster id for each gesture

# TODO I can make this more clever -- gesture cluster gestures
# have gesture_cluster_id in them, match those?
def get_cluster_id_for_gesture(gsm, g_id):
    for k in gsm.GestureClusterer.clusters:
        g_ids = [g['id'] for g in gsm.GestureClusterer.clusters[k]['gestures']]
        if g_id in g_ids:
            return k


def get_gesture_cluster_ids_for_sentence_clusters(gsm):
    for k in gsm.SentenceClusterer.clusters:
        g_cluster_ids = get_gesture_clusters_for_sentence_cluster(k)
        gsm.SentenceClusterer.clusters[k]['gesture_cluster_ids'] = g_cluster_ids

#
# def print_sentences_by_cluster(GSM, cluster_id):
#     sents = GSM.get_sentences_by_cluster(cluster_id)
#     empties = 0
#     c = 0
#     colors = ['red', 'blue']
#     for i, s in enumerate(sents):
#         if s:
#             print colored("%s. %s" % (i, s), colors[c])
#             if c:
#                 c = 0
#             else:
#                 c = 1
#         else:
#             empties += 1
#     print "Along with %s empty strings." % empties
#     print
#
#
#
# def get_sentence_clusters_by_gesture_clusters(GSM):
#     if(GSM.gesture_sentence_clusters):
#         return GSM.gesture_sentence_clusters
#     for k in GSM.GestureClusterer.clusters:
#         g_ids = [g['id'] for g in GSM.GestureClusterer.clusters[k]['gestures']]
#         gests = GSM.get_gestures_by_ids(g_ids)
#         GSM.gesture_sentence_clusters[k] = GSM.SentenceClusterer._create_new_cluster_by_gestures(gests)
