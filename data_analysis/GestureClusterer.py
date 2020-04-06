
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
import timeit
import statistics

from common_helpers import *


BASE_KEYPOINT = [0]
RIGHT_BODY_KEYPOINTS = [1, 2, 3, 28]
LEFT_BODY_KEYPOINTS = [4, 5, 6, 7]
RIGHT_WRIST_KEYPOINT = 3
LEFT_WRIST_KEYPOINT = 6
LEFT_HAND_KEYPOINTS = lambda x: [7] + [8 + (x * 4) + j for j in range(4)]
RIGHT_HAND_KEYPOINTS = lambda x: [28] + [29 + (x * 4) + j for j in range(4)]
ALL_RIGHT_HAND_KEYPOINTS = [3] + list(range(31, 52))
ALL_LEFT_HAND_KEYPOINTS = [6] + list(range(10, 31))
BODY_KEYPOINTS = RIGHT_BODY_KEYPOINTS + LEFT_BODY_KEYPOINTS

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
## Now since we're only working within a single gesture, don't need to worry about
## Mismatched timings anymore.

## see which feature is most influential in clustering

## TODO limit number of clusters
## check distance for clusters with only 1
# https://medium.com/pythonhive/python-decorator-to-measure-the-execution-time-of-methods-fa04cb6bb36d

cluster_method_timings = {'test': 0}

def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            n = method.__name__
            if n in cluster_method_timings.keys():
                cluster_method_timings[n] = cluster_method_timings[n] + (te-ts) * 1000
            else:
                cluster_method_timings[n] = (te-ts) * 1000
            #print('%r  %2.2f ms' % \
            #      (method.__name__, (te - ts) * 1000))
        return result
    return timed





###############################################################
####################### DIY Clustering ########################
###############################################################
class GestureClusterer():
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

    '''
    def cluster_gestures_disparate_seeds(self, gesture_data=None, max_cluster_distance=0.03, max_number_clusters=250):
        gd = gesture_data if gesture_data else self.agd
        if 'feature_vec' in list(gd[0].keys()):
            print("already have feature vectors in our gesture data")
        if not self.has_assigned_feature_vecs and 'feature_vec' not in list(gd[0].keys()):
            self._assign_feature_vectors()

        # randomly sample for now, even though we know that's not the best way to do it at all.
        gestures = random.sample(gd, max_number_clusters)
        random_ids = [g['id'] for g in gestures]
        self.cluster_gestures(gd, max_cluster_distance, max_number_clusters, random_ids)
    '''

    @timeit
    def cluster_gestures(self, gesture_data=None, max_cluster_distance=1200, max_number_clusters=0, seed_ids=[]):
        gd = gesture_data if gesture_data else self.agd
        if 'feature_vec' in list(gd[0].keys()):
            print("already have feature vectors in our gesture data")
        if not self.has_assigned_feature_vecs and 'feature_vec' not in list(gd[0].keys()):
            self._assign_feature_vectors()

        # if we're seeding our clusters with specific gestures
        if len(seed_ids):
            gs = [gesture for gesture in gd if gesture['id'] in seed_ids]
            for g in gs:
                self._create_new_cluster(g)
        self._cluster_gestures(gd, max_cluster_distance, max_number_clusters, seed_ids)

    @timeit
    def _cluster_gestures(self, gd, max_cluster_distance=0.03, max_number_clusters=0, seed_ids=[]):
        print("Clustering gestures")
        avg_min_dist = []
        for g in tqdm(gd):
            # if we've already seeded a cluster with this gesture, don't recluster it.
            if g['id'] in seed_ids:
                continue
            (nearest_cluster_id, nearest_cluster_dist) = self._get_shortest_cluster_dist(g)
            avg_min_dist.append(nearest_cluster_dist)
            if max_number_clusters and len(self.clusters) > max_number_clusters:
                self._add_gesture_to_cluster(g, nearest_cluster_id)
            # we're further away than we're allowed to be, OR this is the first cluster.
            elif (max_cluster_distance and nearest_cluster_dist > max_cluster_distance) or (not len(self.clusters)):
                self._create_new_cluster(g)
                g['cluster_id'] = self.c_id
            else:
                self._add_gesture_to_cluster(g, nearest_cluster_id)
        # self._recluster_singletons()

        # now recluster based on where the new centroids are
        self._recluster_by_centroids()
        print("created %s clusters" % self.total_clusters_created)
        print("avg dist: %s" % statistics.mean(avg_min_dist))
        print("stdev dist: %s" % statistics.stdev(avg_min_dist))


    def _recluster_singletons(self):
        print("reclustering singletons")
        for k in list(self.clusters.keys()):
            if len(self.clusters[k]['gestures']) == 1:
                g = self.clusters[k]['gestures'][0]
                (new_k, dist) = self._get_shortest_cluster_dist(g)
                self._add_gesture_to_cluster(g, new_k)
                del self.clusters[k]

    @timeit
    def _add_gesture_to_cluster(self, g, cluster_id):
        try:
            self.clusters[cluster_id]['gestures'].append(g)
            self._update_cluster_centroid(cluster_id)
            g['gesture_cluster_id'] = cluster_id
        except:
            print('could not add gesture %s to cluster %s' % (g['id'], cluster_id))
            print('cluster keys:')
            print(self.clusters[cluster_id].keys())

    @timeit
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

    @timeit
    def _normalize_feature_values(self, gesture_data=None):
        print("Normalizing feature vectors.")
        gd = gesture_data if gesture_data else self.agd
        feat_vecs = np.array([g['feature_vec'] for g in gd])
        feat_vecs_normalized = self._normalize_across_features(feat_vecs)
        print("Reassigning normalized vectors")
        for i in list(range(len(gd))):
            gd[i]['feature_vec'] = list(feat_vecs_normalized[i])
        return gd

    # takes vectors, normalizes across features
    def _normalize_across_features(self, vectors):
        T = vectors.T
        norms = np.array([v / np.linalg.norm(v) for v in T])
        v_normed = norms.T
        return v_normed

    @timeit
    def _create_new_cluster(self, seed_gest):
        self._log("creating new cluster for gesture %s" % seed_gest['id'])
        new_cluster_id = self.c_id
        self.c_id = self.c_id + 1
        c = {'cluster_id': new_cluster_id,
             'centroid': seed_gest['feature_vec'],
             'seed_id': seed_gest['id'],
             'gestures': [seed_gest]}
        self.clusters[new_cluster_id] = c
        self.total_clusters_created += 1

    # now that we've done the clustering, recluster and only allow clusters to form around current centroids.
    def _recluster_by_centroids(self):
        gd = self.agd
        print("Reclustering by centroid")
        # clear old gestures
        randc = 0
        for c in self.clusters:
            self.clusters[c]['gestures'] = []
            randc = c

        for g in tqdm(gd):
            min_dist = 1000
            min_clust = self.clusters[randc]
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
        print(("Number of clusters: %s" % len(self.clusters)))
        cluster_rep = [(c, len(self.clusters[c]['gestures'])) for c in list(self.clusters.keys())]
        cluster_lengths = [len(self.clusters[c]['gestures']) for c in list(self.clusters.keys())]
        print(("Cluster lengths: %s" % cluster_rep))
        print(("Avg cluster size: %s" % np.average(cluster_lengths)))
        print(("Median cluster size: %s" % np.median(cluster_lengths)))
        print(("Largest cluster size: %s" % max(cluster_lengths)))
        cluster_sparsity = [self.get_cluster_sparsity(c) for c in list(self.clusters.keys())]
        print(("Cluster sparsities: %s" % cluster_sparsity))
        print(("Avg cluster sparsity: %s" % np.average(cluster_sparsity)))
        print(("Median cluster sparsity: %s" % np.median(cluster_sparsity)))
        print(("Sanity check: total clustered gestures: %s / %s" % (sum(cluster_lengths), len(self.agd))))
        print("silhouette score: %s" % self.get_avg_silhouette_score())
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
    @timeit
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
        c['centroid'] = [np.average(x) for x in feat_vecs.T]
        self._log("new centroid: %s" % c['centroid'])
        self.clusters[cluster_id] = c
        e = time.time()
        self._log("time to update centroid: %s" % str(e-s))
        return

    ############################################################
    #################### MOVEMENT CHARACTERISTICS ##############
    ############################################################


    # returns minimum distance at any frame between point A on right hand and
    # point A on left hand.
    def _min_hands_together(self, r_hand_keys, l_hand_keys):
        return self._hand_togetherness(r_hand_keys, l_hand_keys, 10000, min)

    # returns maximum distance at any frame between point A on right hand and
    # point A on left hand
    def _max_hands_apart(self, r_hand_keys, l_hand_keys):
        return self._hand_togetherness(r_hand_keys, l_hand_keys, 0, max)

    @timeit
    def _hand_togetherness(self, r_hand_keys, l_hand_keys, min_max, relate):
        max_dist = min_max          # larger than pixel range
        for frame_index in range(len(r_hand_keys)):     # for each frame in the gesture, better be same cause it's the same gesture!
            # determine how far apart r and l hand are, vertically and horizontally
            cur_r_keys = r_hand_keys[frame_index]
            cur_l_keys = l_hand_keys[frame_index]
            r_pos = np.array((cur_r_keys['x'], cur_r_keys['y']))
            l_pos = np.array((cur_l_keys['x'], cur_l_keys['y']))
            dist = np.linalg.norm(r_pos-l_pos)
            max_dist = relate(dist, max_dist)
        return max_dist

    # get maximum "verticalness" aka minimum horizontalness of hands
    def _palm_vert(self, keyframes):
        return self._palm_angle_axis(keyframes, 'x')

    # get maximum "horizontalness" aka minimum verticalness of hands
    def _palm_horiz(self, keyframes):
        return self._palm_angle_axis(keyframes, 'y')

    @timeit
    def _palm_angle_axis(self, keyframes, xy):
        p_min = 1000
        for frame in keyframes:
            max_frame_dist = max(frame[xy]) - min(frame[xy])
            p_min = min(p_min, max_frame_dist)
        return p_min

    ## max distance from low --> high the wrists move in a single stroke
    def _wrists_up(self, keyframes):
        return self._wrist_vertical_stroke(keyframes, operator.ge)

    ## max distance from high --> low the wrists move in single stroke
    def _wrists_down(self, keyframes):
        return self._wrist_vertical_stroke(keyframes, operator.le)

    @timeit
    def _wrist_vertical_stroke(self, keyframes, relate):
        total_motion = 0
        max_single_stroke = 0
        # 0 is the index of the wrist in handed keypoints
        pos = keyframes[0]['y'][0]
        same_direction = False
        for frame in keyframes:
            curr_pos = frame['y'][0]
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

    # measures the largest outward motion of r/l wrists
    # that is, the largest distance in which wrists are moving
    # continuously apart.
    # THIS IS IN SPACE, USES BOTH HORIZ AND VERT AXES
    def _wrists_outward(self, r_hand_keys, l_hand_keys):
        return self._wrist_relational_move_horizonal(r_hand_keys, l_hand_keys, operator.ge)

    def _wrists_inward(self, r_hand_keys, l_hand_keys):
        return self._wrist_relational_move_horizonal(r_hand_keys, l_hand_keys, operator.le)

    @timeit
    def _wrist_relational_move_horizonal(self, r_hand_keys, l_hand_keys, relate):
        total_direction_dist = 0
        max_direction_dist = 0
        # the 0th keyframe of each hand is the wrist position
        r_wrist_position = np.array([r_hand_keys[0]['x'][0], r_hand_keys[0]['y'][0]])
        l_wrist_position = np.array([l_hand_keys[0]['x'][0], l_hand_keys[0]['y'][0]])
        prev_dist = np.linalg.norm(r_wrist_position - l_wrist_position)
        for frame_index in range(len(r_hand_keys)):
            r_wrist_position = np.array([r_hand_keys[frame_index]['x'][0], r_hand_keys[frame_index]['y'][0]])
            l_wrist_position = np.array([l_hand_keys[frame_index]['x'][0], l_hand_keys[frame_index]['y'][0]])
            cur_dist = np.linalg.norm(r_wrist_position - l_wrist_position)
            if relate(cur_dist, prev_dist):     # we are moving in the desired direction
                total_direction_dist += abs(cur_dist - prev_dist)
                prev_dist = cur_dist
                max_direction_dist = max(max_direction_dist, total_direction_dist)
            else:                       # we're not moving in the desired direction
                total_direction_dist = 0
                prev_dist = cur_dist
        return max_direction_dist


        # there should never be only one frame in a "gesture"
        moving_desired_direction = False
        total_direction_dist = 0
        max_direction_dist = 0
        # r_hand[0] is the first frame, r_hand[0]['x'][0] is the x position of the right wrist in the first frame.
        dist = abs(r_hand[0]['x'][0] - l_hand[0]['x'][0])
        for i in range(1, len(r_hand)):     # r and l hands better be the same length.
            curr_dist = abs(r_hand[i]['x'][0] - l_hand[i]['x'][0])
            if relate(curr_dist, dist):
                moving_desired_direction = True
                total_direction_dist = total_direction_dist + abs(curr_dist - dist)
                max_direction_dist = max(max_direction_dist, total_direction_dist)
                dist = curr_dist
            else:
                if moving_desired_direction:
                    total_direction_dist = 0
                moving_desired_direction = False
                dist = curr_dist
            max_direction_dist = max(max_direction_dist, total_direction_dist)
        return max_direction_dist

    @timeit
    def _wrists_moving_apart(self, r_hand, l_hand):
        # there should never be only one frame in a "gesture"
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
    @timeit
    def _max_wrist_velocity(self, keyframes, num_frames=5):
        max_frame_diff = 0
        for i in range(num_frames, len(keyframes)-1):
            frame_diff = 0
            for j in range(1, num_frames):
                pos1 = self._get_hand_pos(keyframes[i-(j-1)])
                pos2 = self._get_hand_pos(keyframes[i-j])
                frame_diff = frame_diff + self._get_point_dist(pos1[0], pos1[1], pos2[0], pos2[1])
            max_frame_diff = max(max_frame_diff, frame_diff)
        return max_frame_diff

    @timeit
    def _get_hand_pos(self, hand_keys):
        return(self._avg(hand_keys['x']), self._avg(hand_keys['y']))

    @timeit
    def _get_point_dist(self, x1,y1,x2,y2):
        a = np.array((x1, y1))
        b = np.array((x2, y2))
        return np.linalg.norm(a-b)


    ## across all the frames, how much does it go back and forth?
    ## basically, how much do movements switch direction? but on a big scale.
    ## average the amount over the hands
    # def self, _oscillation(gesture):
    #     return

    @timeit
    def _get_gesture_features(self, gesture):
        r_keyframes = self._get_rl_hand_keypoints(gesture, 'r')
        l_keyframes = self._get_rl_hand_keypoints(gesture, 'l')
        gesture_features = [
          self._palm_vert(l_keyframes),
          self._palm_horiz(l_keyframes),
          self._palm_vert(r_keyframes),
          self._palm_horiz(r_keyframes),
          self._max_hands_apart(r_keyframes, l_keyframes),
          self._min_hands_together(r_keyframes, l_keyframes),
          #   x_oscillate,
          #   y_oscillate,
          self._wrists_up(r_keyframes),
          self._wrists_up(l_keyframes),
          self._wrists_down(r_keyframes),
          self._wrists_down(l_keyframes),
          self._wrists_outward(r_keyframes, l_keyframes),
          self._wrists_inward(r_keyframes, l_keyframes),
          self._max_wrist_velocity(r_keyframes),
          self._max_wrist_velocity(l_keyframes)
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
    @timeit
    def _avg(self, v):
        return float(sum(v) / len(v))

    @timeit
    def _get_rl_hand_keypoints(self, gesture, hand):
        keys = []
        keypoint_range = ALL_RIGHT_HAND_KEYPOINTS if hand == 'r' else ALL_LEFT_HAND_KEYPOINTS
        if not gesture['keyframes']:
            print("No keyframes found for gesture")
            print(gesture)
            return

        for t in gesture['keyframes']:
            if (type(t) != dict):
                print("found empty keyframes for gesture %s" % gesture['id'])
                print(t)
                print(gesture)
                self.drop_ids.append(gesture['id'])
                ## KNOWN TEMP FIX
                return [{'y':[0], 'x':[0]}]
            y = [t['y'][i] for i in keypoint_range]
            x = [t['x'][i] for i in keypoint_range]
            keys.append({'y': y, 'x': x})
        return keys

    @timeit
    def _calculate_distance_between_gestures(self, g1, g2):
        if 'feature_vec' in list(g1.keys()) and 'feature_vec' in list(g2.keys()):
            return np.linalg.norm(np.array(g1['feature_vec']) - np.array(g2['feature_vec']))

        feat1 = np.array(self._get_gesture_features(g1))
        feat2 = np.array(self._get_gesture_features(g2))
        return np.linalg.norm(feat1-feat2)

    @timeit
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

    ## a random helper for me to find centroids
    ## and peep average distances
    def _calc_dist_between_random_gestures(self):
        i = random.randrange(0, len(self.agd))
        j = random.randrange(0, len(self.agd))
        return self._calculate_distance_between_vectors(self.agd[i]['feature_vec'], self.agd[j]['feature_vec'])


    ## lol this is wrong.
    ## goes through each gesture twice (which actually might calculate right value but is wrong theoretically)
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
        min_c = 0
        for c in self.clusters:
            if c == cluster_id:
                continue
            mind = self._calculate_distance_between_vectors(self.clusters[c]['centroid'], self.clusters[cluster_id]['centroid'])
            if mind < dist:
                dist = mind
                min_c = c
        return min_c

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
        if len(c['gestures']) == 0:
            print("WARNING: NO GESTURES FOUND IN CLUSTER ID %s" % cluster_id)
            print("num clusters: %s" % len(self.clusters))
            return 0
        for g in c['gestures']:
            dists.append(self._calculate_distance_between_vectors(vec, g['feature_vec']))
        return self._avg(dists)

    # gets silhouette score for cluster using centroid
    def get_silhouette_score(self, cluster_id):
        p = self.clusters[cluster_id]['centroid']
        a = self.get_avg_dist_between_point_and_cluster(p, cluster_id)
        b = self.get_avg_dist_between_point_and_cluster(p, self.get_nearest_cluster_id(cluster_id))
        score = (b - a)/max(b,a)
        return score

    def get_avg_silhouette_score(self):
        scores = []
        for g in self.clusters:
            scores.append(self.get_silhouette_score(g))
        return self._avg(scores)


# our basic problem is that we need to figure out how to map distances between motions that are very long
# vectors, and different lengths of keyframes. But we need to distinguish between the speed of those motions
# as well...

# Another big issue is that individuals get clustered together BECAUSE their large-scale movements are
# similar. But this might not be so much of an issue... if we find patterns that are common to a movement pattern,
# then it's just a case of a gesture cluster representing a personality that expresses a particular trait.


# Silhouette scores for clusters are a good way of determining how many "base" gestures there may be??


