#!/usr/bin/env pythons
from SpeakerGestureGetter import *
from GestureClusterer import *
from common_helpers import *


## the following commands assume you have a full transcript in the cloud
## and also all the timings.
# from GestureSentenceManager import *
# GSM = GestureSentenceManager("/Users/carolynsaund/github/gest-data/data", "rock")
# GSM.load_gestures()
# GSM.cluster_gestures()

## manages gesture and sentence stuff.
class GestureSentenceManager():
    def __init__(self, base_path, speaker, seeds=[]):
        ## this is where the magic is gonna happen.
        ## get all the gestures
        self.base_path = base_path
        self.speaker = speaker
        self.SpeakerGestures = SpeakerGestureGetter(base_path, speaker)
        # self.Clusterer = GestureClusterer(self.SpeakerGestures.all_gesture_data)
        self.cluster_bucket_name = "%s_clusters" % speaker
        self.full_transcript_bucket = "full_timings_with_transcript_bucket"
        self.gesture_transcript = None
        self.get_transcript()
        # now we have clusters, now need to get the corresponding sentences for those clusters.

    def report_clusters(self):
        self.Clusterer.report_clusters()

    def load_gestures(self):
        self.SpeakerGestures.perform_gesture_analysis()

    def cluster_gestures(self):
        self.Clusterer = GestureClusterer(self.SpeakerGestures.all_gesture_data)
        self.Clusterer.cluster_gestures()

    def print_sentences_by_cluster(self, cluster_id):
        sents = self.get_sentences_by_cluster(cluster_id)
        empties = 0
        for i, s in enumerate(sents):
            if s:
                print "%s. %s" % (i, s)
            else:
                empties += 1
        print "Along with %s empty strings." % empties
        print

    def get_sentences_by_cluster(self, cluster_id):
        self.get_transcript()
        gesture_ids = self.Clusterer.get_gesture_ids_by_cluster(cluster_id)
        p = self.gesture_transcript['phrases']
        sentences = [d['phase']['transcript'] for d in p if d['id'] in gesture_ids]
        return sentences

    def get_transcript(self):
        if self.gesture_transcript:
            return
        fp = "temp.json"
        download_blob(self.full_transcript_bucket,
                      "%s_timings_with_transcript.json" % self.speaker,
                      fp)
        self.gesture_transcript = read_data(fp)

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

    def upload_clusters(self):
        self.Clusterer.write_clusters()
        upload_blob(self.cluster_bucket_name, self.Clusterer.cluster_file, self.cluster_bucket_name)