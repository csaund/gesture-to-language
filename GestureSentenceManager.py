


## manages gesture and sentence stuff.
from SpeakerGestureGetter import *
from GestureClusterer import *


class GestureSentenceManager():
    def __init__(self, base_path, speaker, seeds=[]):
        ## this is where the magic is gonna happen.
        ## get all the gestures
        self.base_path = base_path
        self.speaker = speaker
        self.GestureData = SpeakerGestureGetter(base_path, speaker)
        self.Clusterer = GestureClusterer(self.GestureData.all_gesture_data)
        self.Clusterer.cluster_gestures()
        # now we have clusters, now need to get the corresponding sentences for those clusters.

    def get_sentences_by_cluster(self, cluster_id):
        gesture_ids = self.Clusterer.get_gesture_ids_by_cluster(cluster_id)
        
