
#!/usr/bin/env pythons
print("importing libs for GestureClusterer")
import json
import os
import numpy as np
import operator
import random
import time
from tqdm import tqdm
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.metrics import silhouette_samples, silhouette_score

from common_helpers import *
from sklearn import cluster

# semantics -- wn / tf
# rhetorical
# metaphorical
# sentiment
# prosody
# motion dynamics


# do difference between t1 and t2 for same frame,
# and also difference between p1, p2, p3 etc at same frame
# maybe get r_arm_distance_spread,      l_arm_distance_spread,     r_hand_distance_spread, l_hand_distance_spread
#           [p1_t1 - p2_t1 - p3_t1...], [p1_t1 - p2_t1 - p3_t1...]
#  ds_1 =               [t1                       t1...]
#           [p1_t2 - p2_t2 - p3_t2...], [p1_t2 - p2_t2 - p3_t2...]
#  ds_2 =               [t2                       t2...]
# as represending [p1, p2, p3] etc at t1.
# meanwhile each individual point also has a difference
# [p0-p1], [p1-p2]
#    t1      t2
# p0-1_1 = [p0 - p1]
# p0-1_2 = [p1 - p2]
# so for each t you have
# [ds_1, p0-1_1, p1-1_1, p2-1_1, p3-1_1...]
# need to figure out how to compare subsequences of gesture.

## the above is wayyyyy too complicated for now.
## just get a gesture of high-level features, I think
# All of the following refer to if it happens EVER in the gesture
# [
#   max_vert_accel,
#   max_horiz_accel,
#   l_palm_vert,
#   l_palm_horiz,
#   r_palm_vert,
#   r_palm_horiz,
#   x_oscillate,    # does it go up and down a lot?
#   y_oscillate,
#   hands_together,
#   hands_separate,
#   wrists_up,
#   wrists_down,
#   wrists_outward,
#   wrists_inward,
#   wrists_sweep,
#   wrist_arc,
#   r_hand_rotate,
#   l_hand_rotate,
#   hands_cycle
# ]

## see which feature is most influential in clustering
class GestureAnalyzer():
    # all the gesture data for gestures we want to cluster.
    # the ids of any seed gestures we want to use for our clusters.
    def __init__(self, all_gesture_data):
        # I have no idea what best practices are but I'm almost certain this is
        # a gross, disgusting anti-pattern for iterating IDs.
        self.c_id = 0
        self.agd = all_gesture_data
        self.clusters = {}
        self.clf = NearestCentroid()
        self.logs = []
        # todo make this variable
        homePath = os.getenv("HOME")
        self.logfile = os.path.join(homePath, "GestureData", "cluster_logs.txt")
        self.cluster_file = os.path.join(homePath, "GestureData", "cluster_tmp.json")
        self.has_assigned_feature_vecs = False
        self.total_clusters_created = 0
        ## hacky way to work around some malformatted data.
        self.drop_ids = []

    def clear_clusters(self):
        self.clusters = {}
        self.total_clusters_created = 0
        self.c_id = 0


    def cluster_gestures_kmeans(self, k=15, gesture_data=None):
        gd = gesture_data if gesture_data else self.agd
        self.check_feature_vecs()
        X = [g['feature_vec'] for g in gd]
        self.clusterer = cluster.KMeans(n_clusters=k, random_state=10)
        self.cluster_labels = self.clusterer.fit_predict(X)
        self.clusters = {k:[] for k in self.cluster_labels}
        for i in range(0, len(gd)):
            gd[i]['gesture_cluster_id'] = self.cluster_labels[i]         # these sure as heck should be the same length
            self.clusters[self.cluster_labels[i]].append(gd['id'])
        self.agd = gd


    def cluster_gestures_affinity_prop(self, gesture_data=None):
        gd = gesture_data if gesture_data else self.agd
        X = [g['feature_vec'] for g in gd]
        self.cluster_labels = cluster.AffinityPropagation().fit_predict(X)
        self.clusters = {k:[] for k in self.cluster_labels}
        for i in range(0, len(gd)):
            gd[i]['gesture_cluster_id'] = self.cluster_labels[i]         # these sure as heck should be the same length
            self.clusters[self.cluster_labels[i]].append(gd['id'])
        self.agd = gd


    def get_silhouette(self, n_clusters):
        X = [g['feature_vec'] for g in gd]
        silhouette_avg = silhouette_score(X, self.cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

    def check_feature_vecs(self):
        if 'feature_vec' in list(gd[0].keys()):
            print("already have feature vectors in our gesture data")
        if not self.has_assigned_feature_vecs and 'feature_vec' not in list(gd[0].keys()):
            self._assign_feature_vectors()


    def _assign_feature_vectors(self, gesture_data=None):
        gd = gesture_data if gesture_data else self.agd
        empty_vec = self._create_empty_feature_vector()

        print("Getting initial feature vectors.")
        for g in tqdm(gd):
            if type(g['keyframes']) == type(None):
                print("found empty vector")
                ## TODO fix this
                g['feature_vec'] = empty_vec
            else:
                g['feature_vec'] = self._get_gesture_features(g)

        empties = [g for g in self.agd if g['feature_vec'] == empty_vec]
        print("dropping %s empty vectors from gesture clusters" % str(len(empties)))
        # hacky ways to fix malformatted data
        self.agd = [g for g in self.agd if g['feature_vec'] != empty_vec]
        self.agd = [g for g in self.agd if g['id'] not in self.drop_ids]
        self.drop_ids = list(set(self.drop_ids))
        self._normalize_feature_values()
        self.has_assigned_feature_vecs = True
        return


    def _normalize_feature_values(self, gesture_data=None):
        print("Normalizing feature vectors.")
        gd = gesture_data if gesture_data else self.agd
        feat_vecs = np.array([g['feature_vec'] for g in self.agd])
        feats_norm = np.array([v / np.linalg.norm(v) for v in feat_vecs.T])
        feat_vecs_normalized = feats_norm.T
        print("Reassigning normalized vectors")
        for i in tqdm(list(range(len(gd)))):
            gd[i]['feature_vec'] = list(feat_vecs_normalized[i])
        return


    def get_sentences_by_cluster(self, cluster_id):
        c = self.clusters[cluster_id]
        sents = [g['phase']['transcript'] for g in c['gestures']]
        return sents


    def get_gesture_ids_by_cluster(self, cluster_id):
        c = self.clusters[cluster_id]
        ids = [g['id'] for g in c]
        return ids

    ############################################################
    #################### MOVEMENT CHARACTERISTICS ##############
    ############################################################
    # returns minimum distance at any frame between point A on right hand and
    # point A on left hand.
    def _min_hands_together(self, gesture):
        return self._hand_togetherness(gesture, 1000, min)


    # returns maximum distance at any frame between point A on right hand and
    # point A on left hand
    def _max_hands_apart(self, gesture):
        return self._hand_togetherness(gesture, 0, max)


    def _hand_togetherness(self, gesture, min_max, relate):
        r_hand_keys = self._get_rl_hand_keypoints(gesture, 'r')
        l_hand_keys = self._get_rl_hand_keypoints(gesture, 'l')
        max_dist = min_max # larger than pixel range
        for i in range(0, len(r_hand_keys)-2):
            for j in range(0, len(r_hand_keys[i]['x'])-1):
                r_x = r_hand_keys[i]['x'][j]
                r_y = r_hand_keys[i]['y'][j]
                l_x = l_hand_keys[i]['x'][j]
                l_y = l_hand_keys[i]['y'][j]
                a = np.array((r_x, r_y))
                b = np.array((l_x, l_y))
                dist = np.linalg.norm(a-b)
                max_dist = relate(dist, max_dist)
        return max_dist


    # get maximum "verticalness" aka minimum horizontalness of hands
    def _palm_vert(self, gesture, lr):
        return self._palm_angle_axis(gesture, lr, 'x')


    # get maximum "horizontalness" aka minimum verticalness of hands
    def _palm_horiz(self, gesture, lr):
        return self._palm_angle_axis(gesture, lr, 'y')


    def _palm_angle_axis(self, gesture, lr, xy):
        hand_keys = self._get_rl_hand_keypoints(gesture, lr)
        p_min = 1000
        for frame in hand_keys:
            max_frame_dist = max(frame[xy]) - min(frame[xy])
            p_min = min(p_min, max_frame_dist)
        return p_min


    ## max distance from low --> high the wrists move in a single stroke
    def _wrists_up(self, gesture, lr):
        return self._wrist_vertical_stroke(gesture, lr, operator.ge)


    ## max distance from high --> low the wrists move in single stroke
    def _wrists_down(self, gesture, lr):
        return self._wrist_vertical_stroke(gesture, lr, operator.le)


    def _wrist_vertical_stroke(self, gesture, lr, relate):
        hand_keys = self._get_rl_hand_keypoints(gesture, lr)
        total_motion = 0
        max_single_stroke = 0
        pos = self._avg(hand_keys[0]['y'])
        same_direction = False
        for frame in hand_keys:
            curr_pos = self._avg(frame['y'])
            if relate(curr_pos, pos):
                total_motion = total_motion + abs(curr_pos - pos)
                pos = curr_pos
                same_direction = True
            else:
                if same_direction:
                    total_motion = 0
                same_direction = False
                pos = curr_pos
            max_single_stroke = max(max_single_stroke, total_motion)
        return max_single_stroke


    def _wrists_outward(self, gesture):
        return self._wrist_relational_move(gesture, operator.ge)


    def _wrists_inward(self, gesture):
        return self._wrist_relational_move(gesture, operator.le)


    def _wrist_relational_move(self, gesture, relate):
        r_hand = self._get_rl_hand_keypoints(gesture, 'r')
        l_hand = self._get_rl_hand_keypoints(gesture, 'l')
        moving_desired_direction = False
        total_direction_dist = 0
        max_direction_dist = 0
        dist = abs(self._avg(r_hand[0]['x']) - self._avg(l_hand[0]['x']))
        for i in range(1, len(r_hand)-1):
            curr_dist = abs(self._avg(r_hand[i]['x']) - self._avg(l_hand[i]['x']))
            if relate(curr_dist, dist):
                moving_desired_direction = True
                total_direction_dist = total_direction_dist + abs(curr_dist - dist)
                dist = curr_dist
            else:
                if moving_desired_direction:
                    total_direction_dist = 0
                moving_desired_direction = False
                dist = curr_dist
            max_direction_dist = max(max_direction_dist, total_direction_dist)
        return max_direction_dist


    def _wrists_moving_apart(self, gesture):
        r_hand = self._get_rl_hand_keypoints(gesture, 'r')
        l_hand = self._get_rl_hand_keypoints(gesture, 'l')
        moving_apart = False
        total_apart = 0
        max_dist_apart = 0
        dist = get_point_dist(self._avg(r_hand[0]['x']), self._avg(r_hand[0]['y']), self._avg(l_hand[0]['x']), self._avg(l_hand[0]['y']))
        for i in range(1, len(r_hand)-1):
            curr_dist = get_point_dist(self._avg(r_hand[i]['x']), self._avg(r_hand[i]['y']), self._avg(l_hand[i]['x']), self._avg(l_hand[i]['y']))
            if curr_dist >= dist:
                moving_apart = True
                total_apart = total_apart + abs(curr_dist - dist)
                dist = curr_dist
            else:
                if moving_apart:
                    total_apart = 0
                moving_inward = False
                dist = curr_dist
            max_dist_apart = max(max_dist_apart, total_apart)
        return max_dist_apart


    # max velocity over n frames.
    # only goes over r/l hand avg pos
    def _max_wrist_velocity(self, gesture, rl='r', num_frames=5):
        hand_keys = self._get_rl_hand_keypoints(gesture, rl)
        max_frame_diff = 0
        for i in range(num_frames, len(hand_keys)-1):
            frame_diff = 0
            for j in range(1, num_frames):
                pos1 = self._get_hand_pos(hand_keys[i-(j-1)])
                pos2 = self._get_hand_pos(hand_keys[i-j])
                frame_diff = frame_diff + self._get_point_dist(pos1[0], pos1[1], pos2[0], pos2[1])
            max_frame_diff = max(max_frame_diff, frame_diff)
        return max_frame_diff


    def _get_hand_pos(self, hand_keys):
        return(self._avg(hand_keys['x']), self._avg(hand_keys['y']))


    def _get_point_dist(self, x1,y1,x2,y2):
        a = np.array((x1, y1))
        b = np.array((x2, y2))
        return np.linalg.norm(a-b)


    ## across all the frames, how much does it go back and forth?
    ## basically, how much do movements switch direction? but on a big scale.
    ## average the amount over the hands
    # def self, _oscillation(gesture):
    #     return


    def _get_gesture_features(self, gesture):
        gesture_features = [
          self._palm_vert(gesture, 'l'),
          self._palm_horiz(gesture, 'l'),
          self._palm_vert(gesture, 'r'),
          self._palm_horiz(gesture, 'r'),
          self._max_hands_apart(gesture),
          self._min_hands_together(gesture),
          #   x_oscillate,
          #   y_oscillate,
          self._wrists_up(gesture, 'r'),
          self._wrists_up(gesture, 'l'),
          self._wrists_down(gesture, 'r'),
          self._wrists_down(gesture, 'l'),
          self._wrists_outward(gesture),
          self._wrists_inward(gesture),
          self._max_wrist_velocity(gesture, 'r'),
          self._max_wrist_velocity(gesture, 'l')
        #   wrists_sweep,
        #   wrist_arc,
        #   r_hand_rotate,
        #   l_hand_rotate,
        #   hands_cycle
        ]
        return gesture_features

    def _create_empty_feature_vector(self):
        gesture_features = [0,0,0,0,0,0,0,0,0,0,0,0]
        return gesture_features

    # TODO manually make some features more/less important
    # report importance of each individual feature in clustering
    # try seeding with some gestures? -- bias, but since we're doing it based on prior gesture research it's ok

    # TODO check what audio features are in original paper

    ## Audio for agent is bad -- TTS is garbaggio
    ## assumes there is good prosody in voice (TTS there isn't)

    ## Co-optimize gesture-language clustering (learn how)
    ## KL distance for two clusters?

    # learn similarity of sentences from within one gesture
    # how to map gesture clusters <--> sentence clusters
    ## in the end want to optimize overlapping clusters btw gesture/language

    # probabilistic mapping of sentence (from gesture) to sentence cluster

    ##############################################################
    ####################### Helpers/Calcs ########################
    ##############################################################
    def _avg(self, v):
        return float(sum(v) / len(v))


    def _get_body_keypoints(self, gesture):
        return self._get_keypoints_body_range(gesture, 0, 7)


    def _get_rl_hand_keypoints(self, gesture, hand):
        if hand == 'r':
            return self._get_keypoints_body_range(gesture, 7, 28)
        elif hand == 'l':
            return self._get_keypoints_body_range(gesture, 28, 49)


    def _get_keypoints_body_range(self, gesture, start, end):
        keys = []
        if not gesture['keyframes']:
            #
            print("OH NO")
            print(gesture)
            return

        for t in gesture['keyframes']:
            if (type(t) != dict):
                print(gesture['id'])
                print("T IS NOT A DICT???")
                print(t)
                print(gesture)
                self.drop_ids.append(gesture['id'])
                ## KNOWN TEMP FIX
                return [{'y': [247, 242, 387, 446, 260, 425, 418, 151, 127, 131, 418, 427, 438, 459, 482, 430, 445, 468, 479, 427, 456, 471, 478, 431, 461, 475, 486, 439, 464, 479, 491, 420, 416, 447, 415, 466, 443, 460, 473, 479, 432, 456, 472, 482, 432, 452, 462, 472, 427, 449, 454, 459],
                         'x':[326, 199, 160, 267, 449, 499, 378, 327, 305, 350, 371, 341, 317, 297, 269, 316, 280, 298, 299, 330, 317, 318, 323, 347, 335, 335, 336, 360, 353, 351, 352, 367, 343, 282, 347, 262, 357, 353, 347, 351, 342, 334, 332, 335, 330, 320, 319, 315, 319, 307, 306, 306]}]
            y = t['y'][start:end]
            x = t['x'][start:end]
            keys.append({'y': y, 'x': x})
        return keys


    def _calculate_distance_between_vectors(self, v1, v2):
        return np.linalg.norm(np.array(v1) - np.array(v2))


    def _log(self, s):
        self.logs.append(s)


    def _write_logs(self):
        with open(self.logfile, 'w') as f:
            for l in self.logs:
                f.write("%s\n" % l)
        f.close()


# our basic problem is that we need to figure out how to map distances between motions that are very long
# vectors, and different lengths of keyframes. But we need to distinguish between the speed of those motions
# as well...

# Another big issue is that individuals get clustered together BECAUSE their large-scale movements are
# similar. But this might not be so much of an issue... if we find patterns that are common to a movement pattern,
# then it's just a case of a gesture cluster representing a personality that expresses a particular trait.


# Silhouette scores for clusters are a good way of determining how many "base" gestures there may be??
