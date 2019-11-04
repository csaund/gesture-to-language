from __future__ import division
#!/usr/bin/env pythons
print "importing libs for GestureClusterer"
import argparse
import pandas as pd
import itertools
import csv
import math
import json
import glob, os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import uuid
import operator
import random
import time
from tqdm import tqdm
from sklearn.neighbors.nearest_centroid import NearestCentroid

from common_helpers import *

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
## Now since we're only working within a single gesture, don't need to worry about
## Mismatched timings anymore.

## see which feature is most influential in clustering

## TODO limit number of clusters
## check distance for clusters with only 1

###############################################################
####################### DIY Clustering ########################
###############################################################
class GestureClusterer():
    # all the gesture data for gestures we want to cluster.
    # the ids of any seed gestures we want to use for our clusters.
    def __init__(self, all_gesture_data, seeds=[]):
        # I have no idea what best practices are but I'm almost certain this is
        # a gross, disgusting anti-pattern for iterating IDs.
        self.c_id = 0
        self.agd = all_gesture_data
        self.clusters = {}
        self.seed_ids = seeds
        self.clf = NearestCentroid()
        self.logs = []
        # todo make this variable
        self.logfile = "%s/github/gesture-to-language/cluster_logs.txt" % os.getenv("HOME")
        self.cluster_file = "%s/github/gesture-to-language/cluster_tmp.json" % os.getenv("HOME")
        if(len(seeds)):
            for seed_g in seeds:
                g = self._get_gesture_by_id(seed_g, all_gesture_data)
                cluster_id = self.c_id
                self.c_id = self.c_id + 1
                c = {
                        'cluster_id': cluster_id,
                        'seed_id': g['id'],
                        'centroid': self._get_gesture_features(g),
                        'gestures': [g['id']]}
                self.clusters[cluster_id] = c
        self.has_assigned_feature_vecs = False

    def cluster_gestures(self, gesture_data=None, max_cluster_distance=0.03):
        if not self.has_assigned_feature_vecs:
            self._assign_feature_vectors()
        gd = gesture_data if gesture_data else self.agd
        i = 0
        l = len(gd)
        print "Clustering gestures"
        for g in tqdm(gd):
            start = time.time()
            i = i + 1
            # print("finding cluster for gesture %s (%s/%s)" % (g['id'], i, l))
            self._log("finding cluster for gesture %s (%s/%s)" % (g['id'], i, l))
            (nearest_cluster_id, nearest_cluster_dist) = self._get_shortest_cluster_dist(g)
            # we're further away than we're allowed to be, OR this is the first cluster.
            if (max_cluster_distance and nearest_cluster_dist > max_cluster_distance) or (not len(self.clusters)):
                self._log("nearest cluster distance was %s" % nearest_cluster_dist)
                self._log("creating new cluster for gesture %s -- %s" % (g['id'], i))
                self._create_new_cluster(g)
                g['cluster_id'] = self.c_id
            else:
                self._log("nearest cluster distance was %s" % nearest_cluster_dist)
                self._add_gesture_to_cluster(g, nearest_cluster_id)
            end = time.time()
            self._log(str(end-start))

        self._recluster_singletons()

        # now recluster based on where the new centroids are
        self._recluster_by_centroids()
        self._write_logs()

    def _recluster_singletons(self):
        print "reclustering singletons"
        for k in self.clusters.keys():
            if len(self.clusters[k]['gestures']) == 1:
                g = self.clusters[k]['gestures'][0]
                (new_k, dist) = self._get_shortest_cluster_dist(g)
                self._add_gesture_to_cluster(g, new_k)
                del self.clusters[k]

    def _add_gesture_to_cluster(self, g, cluster_id):
        self._log("adding gesture %s to cluster %s" % (g['id'], cluster_id))
        self.clusters[cluster_id]['gestures'].append(g)
        self._update_cluster_centroid(cluster_id)
        g['gesture_cluster_id'] = cluster_id


    def _assign_feature_vectors(self, gesture_data=None):
        gd = gesture_data if gesture_data else self.agd
        print "Getting initial feature vectors."
        for g in tqdm(gd):
            g['feature_vec'] = self._get_gesture_features(g)
        self._normalize_feature_values()
        self.has_assigned_feature_vecs = True
        return

    def _normalize_feature_values(self, gesture_data=None):
        print "Normalizing feature vectors."
        gd = gesture_data if gesture_data else self.agd
        feat_vecs = np.array([g['feature_vec'] for g in self.agd])
        feats_norm = np.array(map(lambda v: v / np.linalg.norm(v), feat_vecs.T))
        feat_vecs_normalized = feats_norm.T
        print "Reassigning normalized vectors"
        for i in tqdm(range(len(gd))):
            gd[i]['feature_vec'] = list(feat_vecs_normalized[i])
        return


    def _create_new_cluster(self, seed_gest):
        self._log("creating new cluster for gesture %s" % seed_gest['id'])
        new_cluster_id = self.c_id
        self.c_id = self.c_id + 1
        c = {'cluster_id': new_cluster_id,
             'centroid': seed_gest['feature_vec'],
             'seed_id': seed_gest['id'],
             'gestures': [seed_gest]}
        self.clusters[new_cluster_id] = c

    # now that we've done the clustering, recluster and only allow clusters to form around current centroids.
    def _recluster_by_centroids(self):
        gd = self.agd
        print "Reclustering by centroid"
        # clear old gestures
        for c in self.clusters:
            self.clusters[c]['gestures'] = []

        for g in tqdm(gd):
            min_dist = 1000
            min_clust = self.clusters[0]
            for c in self.clusters:
                d = self._calculate_distance_between_vectors(g['feature_vec'], self.clusters[c]['centroid'])
                if d < min_dist:
                    min_dist = d
                    min_clust = self.clusters[c]
            min_clust['gestures'].append(g)
        return

    def get_sentences_by_cluster(self, cluster_id):
        c = self.clusters[cluster_id]
        sents = [g['phase']['transcript'] for g in c['gestures']]
        return sents

    def get_gesture_ids_by_cluster(self, cluster_id):
        c = self.clusters[cluster_id]
        ids = [g['id'] for g in c['gestures']]
        # ids = []
        # for g in c['gestures']:
        #     ids.append(g['id'])
        return ids

    def report_clusters(self, verbose=False):
        print("Number of clusters: %s" % len(self.clusters))
        num_clusters = len(self.clusters)
        cluster_lengths = [len(self.clusters[c]['gestures']) for c in self.clusters.keys()]
        print("Cluster lengths: %s" % cluster_lengths)
        print("Avg cluster size: %s" % np.average(cluster_lengths))
        print("Median cluster size: %s" % np.median(cluster_lengths))
        print("Largest cluster size: %s" % max(cluster_lengths))
        cluster_sparsity = [self.get_cluster_sparsity(c) for c in self.clusters.keys()]
        print("Cluster sparsities: %s" % cluster_sparsity)
        print("Avg cluster sparsity: %s" % np.average(cluster_sparsity))
        print("Median cluster sparsity: %s" % np.median(cluster_sparsity))
        print("Sanity check: total clustered gestures: %s / %s" % (sum(cluster_lengths), len(self.agd)))
        # TODO: average and median centroid distances from each other.
        # TODO: also get minimum and maximum centroid distances.
        return self.clusters


    ## measure of how distant the gestures are... basically avg distance to centroid
    def get_cluster_sparsity(self, cluster_id):
        c = self.clusters[cluster_id]
        cent = c['centroid']
        dists = [self._calculate_distance_between_vectors(g['feature_vec'], cent) for g in c['gestures']]
        return np.average(dists)


    ## instead of this need to use centroid.
    def _get_shortest_cluster_dist(self, g):
        shortest_dist = 10000
        nearest_cluster_id = ''
        for k in self.clusters:
            c = self.clusters[k]
            centroid = c['centroid']
            dist = self._calculate_distance_between_vectors(g['feature_vec'], centroid)
            if dist < shortest_dist:
                nearest_cluster_id = c['cluster_id']
            shortest_dist = min(shortest_dist, dist)
        return (nearest_cluster_id, shortest_dist)

    ## TODO check this bad boy for bugs.
    def _update_cluster_centroid(self, cluster_id):
        s = time.time()
        c = self.clusters[cluster_id]
        ## very very slow.
        ## TODO speed this up using matrix magic.
        feat_vecs = [g['feature_vec'] for g in c['gestures']]
        feat_vecs = np.array(feat_vecs[0])
        self._log("old centroid: %s" % c['centroid'])
        c['centroid'] = map(lambda x: np.average(x), feat_vecs.T)
        self._log("new centroid: %s" % c['centroid'])
        self.clusters[cluster_id] = c
        e = time.time()
        self._log("time to update centroid: %s" % str(e-s))
        return

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

    # TODO manually make some features more/less important
    # report importance of each individual feature in clustering
    # try seeding with some gestures? -- bias, but since we're doing it based on prior gesture research it's ok

    # TODO check what audio features are in original paper

    ## Audio for agent is bad -- TTS is garbaggio
    ## assumes there is good prosody in voice (TTS there isn't)

    ## Co-optimize gesture-language clustering (learn how)
    ## KL distance for two clusters?

    ## Frame intro of paper

    # learn similarity of sentences from within one gesture
    # how to map gesture clusters <--> sentence clusters
    ## in the end want to optimize overlapping clusters btw gesture/language

    # probabilistic mapping of sentence (from gesture) to sentence cluster

    ########################################################
    ####################### Helpers ########################
    ########################################################
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
        for t in gesture['keyframes']:
            y = t['y'][start:end]
            x = t['x'][start:end]
            keys.append({'y': y, 'x': x})
        return keys

    def _calculate_distance_between_gestures(self, g1, g2):
        feat1 = np.array(self._get_gesture_features(g1))
        feat2 = np.array(self._get_gesture_features(g2))
        return np.linalg.norm(feat1-feat2)

    def _calculate_distance_between_vectors(self, v1, v2):
        return np.linalg.norm(np.array(v1) - np.array(v2))

    def _log(self, s):
        self.logs.append(s)

    def _write_logs(self):
        with open(self.logfile, 'w') as f:
            for l in self.logs:
                f.write("%s\n" % l)
        f.close()

    def get_closest_gesture_to_centroid(self, cluster_id):
        c = self.clusters[cluster_id]
        cent = c['centroid']
        min_d = 1000
        g_id = 0
        for g in c['gestures']:
            dist = self._calculate_distance_between_vectors(g['feature_vec'], cent)
            if dist < min_d:
                g_id = g['id']
                dist = min_d
        return g_id

    def get_random_gesture_id_from_cluster(self, cluster_id):
        c = self.clusters[cluster_id]
        i = random.randrange(0, len(c['gestures']))
        return c['gestures'][i]['id']

    def write_clusters(self):
        with open(self.cluster_file, 'w') as f:
            json.dump(self.clusters, f, indent=4)
        f.close()

    def delete_cluster_file(self):
        os.remove(self.cluster_file)

    def delete_log_file(self):
        os.remove(self.logfile)

    ## a random helper for me to find centroids.
    def _calc_dist_between_random_gestures(self):
        i = random.randrange(0, len(self.agd))
        j = random.randrange(0, len(self.agd))
        return self._calculate_distance_between_vectors(self.agd[i]['feature_vec'], self.agd[j]['feature_vec'])


    def get_avg_within_cluster_distances(self, cluster_id):
        c = self.clusters[cluster_id]
        gs = c['gestures']
        all_dists = []
        for i in range(len(gs)):
            for j in range(len(gs)):
                if i == j:
                    continue
                all_dists = all_dists + self._calculate_distance_between_vectors(gs[i]['feature_vec'], gs[j]['feature_vec'])

        dists = {}
        dists['average'] = self._avg(all_dists)
        dists['max'] = max(all_dists)
        dists['min'] = min(all_dists)


    # takes a cluster ID, returns nearest neighbor cluster ID
    def get_nearest_cluster_id(self, cluster_id):
        dist = 1000
        for c in self.clusters:
            if c == cluster_id:
                continue
            mind = self._calculate_distance_between_vectors(self.clusters[c]['centroid'], self.clusters[cluster_id]['centroid'])
            if mind < dist:
                dist = mind
                min_c = c
        return c

    # takes cluster ID, returns nearest neighbor cluster distance
    def get_nearest_cluster_distance(self, cluster_id):
        dist = 1000
        for c in self.clusters:
            if c == cluster_id:
                continue
            mind = self._calculate_distance_between_vectors(self.clusters[c]['centroid'], self.clusters[cluster_id]['centroid'])
            if mind < dist:
                dist = mind
                min_c = c
        return dist


    # given a point and cluster id, returns avg distance between the point and
    # all points in the cluster.
    def get_avg_dist_between_point_and_cluster(self, vec, cluster_id):
        dists = []
        c = self.clusters[cluster_id]
        for g in c['gestures']:
            dists.append(self._calculate_distance_between_vectors(vec), g['feature_vec'])
        return self._avg(dists)

    # gets silhouette score for cluster using centroid
    def get_silhouette_score(self, cluster_id):
        p = self.clusters[cluster_id]['centroid']
        nearest_c = get_nearest_cluster_id(cluster_id)
        a = self.get_avg_dist_between_point_and_cluster(p, cluster_id)
        b = self.get_avg_dist_between_point_and_cluster(p, self.get_nearest_cluster_id(cluster_id))
        score = (b - a)/max(b,a)
        return score



#
#
# def get_closest_gesture_to_centroid(GSM, cluster_id):
#     c = GSM.Clusterer.clusters[cluster_id]
#     cent = c['centroid']
#     min_d = 1000
#     g_id = 0
#     for g in c['gestures']:
#         dist = GSM.Clusterer._calculate_distance_between_vectors(g['feature_vec'], cent)
#         if dist < min_d:
#             g_id = g['id']
#             dist = min_d
#     return g_id
#
#
# def get_random_gesture_id_from_cluster(GSM, cluster_id):
#     c = GSM.Clusterer.clusters[cluster_id]
#     i = random.randrange(0, len(c['gestures']))
#     return c['gestures'][i]['id']
