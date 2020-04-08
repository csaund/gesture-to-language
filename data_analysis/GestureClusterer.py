#!/usr/bin/env pythons
import json
import os
import numpy as np
import random
import time
from tqdm import tqdm
from sklearn.neighbors.nearest_centroid import NearestCentroid
from GestureMovementHelpers import *
from common_helpers import *


# semantics -- wn / tf
# rhetorical
# metaphorical
# sentiment
# prosody
# motion dynamics

# Now since we're only working within a single gesture, don't need to worry about
# Mismatched timings anymore.

# TODO
# see which feature is most influential in clustering
# add cycle detection to features
# add rotation? arcs? sweeps? oscillation?


###############################################################
# DIY Clustering ##############################################
###############################################################
# takes vectors, normalizes across features
def _normalize_across_features(vectors):
    T = vectors.T
    norms = np.array([v / np.linalg.norm(v) for v in T])
    v_normed = norms.T
    return v_normed


def _create_empty_feature_vector():
    gesture_features = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    return gesture_features


@timeit
def _avg(v):
    return float(sum(v) / len(v))


class GestureClusterer:
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

    def clear_clusters(self):
        self.clusters = {}
        self.total_clusters_created = 0
        self.c_id = 0

    @timeit
    def cluster_gestures(self, gesture_data=None, gesture_features=GESTURE_FEATURES, max_cluster_distance=0.10,
                         max_number_clusters=0, seed_ids=None):
        if seed_ids is None:
            seed_ids = []
        gd = gesture_data if gesture_data else self.agd
        if 'feature_vec' in list(gd[0].keys()):
            print("already have feature vectors in our gesture data")
        if not self.has_assigned_feature_vecs and 'feature_vec' not in list(gd[0].keys()):
            self._assign_feature_vectors(gesture_features=gesture_features)
        # if we're seeding our clusters with specific gestures
        if len(seed_ids):
            gs = [gesture for gesture in gd if gesture['id'] in seed_ids]
            for g in gs:
                self._create_new_cluster(g)

        self._cluster_gestures(gd, max_cluster_distance, max_number_clusters, seed_ids)

    @timeit
    def _cluster_gestures(self, gd, max_cluster_distance=0.03, max_number_clusters=0, seed_ids=None):
        if seed_ids is None:
            seed_ids = []
        print("Clustering gestures")
        for g in tqdm(gd):
            # if we've already seeded a cluster with this gesture, don't cluster it.
            if g['id'] in seed_ids:
                continue
            (nearest_cluster_id, nearest_cluster_dist) = self._get_shortest_cluster_dist(g)
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
        return

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
        except RuntimeError as e:
            print('could not add gesture %s to cluster %s' % (g['id'], cluster_id))
            print('cluster keys:')
            print(e)
            print(self.clusters[cluster_id].keys())

    @timeit
    def _assign_feature_vectors(self, gesture_data=None, gesture_features=GESTURE_FEATURES):
        gd = gesture_data if gesture_data else self.agd
        empty_vec = _create_empty_feature_vector()

        print("Getting initial feature vectors.")
        for g in tqdm(gd):
            if isinstance(g['keyframes'], type(None)):
                print("found empty vector")
                # TODO fix this
                g['feature_vec'] = empty_vec
            else:
                g['feature_vec'] = self._get_gesture_features(g, gesture_features)

        empties = [g for g in self.agd if g['feature_vec'] == empty_vec]
        print("dropping %s empty vectors from gesture clusters" % str(len(empties)))
        # hacky ways to fix malformatted data
        self.agd = [g for g in self.agd if g['feature_vec'] != empty_vec]
        self._normalize_feature_values()
        self.has_assigned_feature_vecs = True
        return

    def get_feature_vector_by_gesture_id(self, g_id):
        g = [i['feature_vec'] for i in self.agd if i['id'] == g_id]
        if len(g):
            g = g[0]
        else:
            g = []
        return g

    @timeit
    def _normalize_feature_values(self, gesture_data=None):
        print("Normalizing feature vectors.")
        gd = gesture_data if gesture_data else self.agd
        feat_vecs = np.array([g['feature_vec'] for g in gd])
        feat_vecs_normalized = _normalize_across_features(feat_vecs)
        print("Reassigning normalized vectors")
        for i in list(range(len(gd))):
            gd[i]['feature_vec'] = list(feat_vecs_normalized[i])
        return gd

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
        rand_c = 0
        for c in self.clusters:
            self.clusters[c]['gestures'] = []
            rand_c = c

        for g in tqdm(gd):
            min_dist = 1000
            min_cluster = self.clusters[rand_c]
            for c in self.clusters:
                d = self._calculate_distance_between_vectors(g['feature_vec'], self.clusters[c]['centroid'])
                if d < min_dist:
                    min_dist = d
                    min_cluster = self.clusters[c]
            min_cluster['gestures'].append(g)
        return

    def get_sentences_by_cluster(self, cluster_id):
        c = self.clusters[cluster_id]
        sentences = [g['phase']['transcript'] for g in c['gestures']]
        return sentences

    def get_gesture_ids_by_cluster(self, cluster_id):
        c = self.clusters[cluster_id]
        ids = [g['id'] for g in c['gestures']]
        # ids = []
        # for g in c['gestures']:
        #     ids.append(g['id'])
        return ids

    def report_clusters(self):
        print(("Number of clusters: %s" % len(self.clusters)))
        cluster_rep = [(c, len(self.clusters[c]['gestures'])) for c in list(self.clusters.keys())]
        cluster_lengths = [len(self.clusters[c]['gestures']) for c in list(self.clusters.keys())]
        print(("Cluster lengths: %s" % cluster_rep))
        print(("Avg cluster size: %s" % np.average(cluster_lengths)))
        print(("Median cluster size: %s" % np.median(cluster_lengths)))
        print(("Largest cluster size: %s" % max(cluster_lengths)))
        cluster_sparsity = [self.get_cluster_sparsity(c) for c in list(self.clusters.keys())]
        print(("Cluster sparsity: %s" % cluster_sparsity))
        print(("Avg cluster sparsity: %s" % np.average(cluster_sparsity)))
        print(("Median cluster sparsity: %s" % np.median(cluster_sparsity)))
        print(("Sanity check: total clustered gestures: %s / %s" % (sum(cluster_lengths), len(self.agd))))
        print("silhouette score: %s" % self.get_avg_silhouette_score())
        # TODO: average and median centroid distances from each other.
        # TODO: also get minimum and maximum centroid distances.
        return self.clusters

    # measure of how distant the gestures are... basically avg distance to centroid
    def get_cluster_sparsity(self, cluster_id):
        c = self.clusters[cluster_id]
        cent = c['centroid']
        dists = [self._calculate_distance_between_vectors(g['feature_vec'], cent) for g in c['gestures']]
        return np.average(dists)

    # instead of this need to use centroid.
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
        return nearest_cluster_id, shortest_dist

    # TODO check this bad boy for bugs.
    def _update_cluster_centroid(self, cluster_id):
        s = time.time()
        c = self.clusters[cluster_id]
        # very very slow.
        # TODO speed this up using matrix magic.
        feat_vecs = [g['feature_vec'] for g in c['gestures']]
        feat_vecs = np.array(feat_vecs[0])
        self._log("old centroid: %s" % c['centroid'])
        c['centroid'] = [np.average(x) for x in feat_vecs.T]
        self._log("new centroid: %s" % c['centroid'])
        self.clusters[cluster_id] = c
        e = time.time()
        self._log("time to update centroid: %s" % str(e - s))
        return

    ############################################################
    # MOVEMENT CHARACTERISTICS #################################
    ############################################################
    @timeit
    def _get_gesture_features(self, gesture, gesture_features=GESTURE_FEATURES):
        r_keyframes = self._get_rl_hand_keypoints(gesture, 'r')
        l_keyframes = self._get_rl_hand_keypoints(gesture, 'l')
        feature_vector = []
        for feature in gesture_features:
            if not GESTURE_FEATURES[feature]['separate_hands']:
                val = GESTURE_FEATURES[feature]['function'](r_keyframes, l_keyframes)
                feature_vector.append(val)
            else:
                # TODO combine all two-handed to just max overall?
                val1 = GESTURE_FEATURES[feature]['function'](r_keyframes)
                val2 = GESTURE_FEATURES[feature]['function'](l_keyframes)
                feature_vector.append(val1)
                feature_vector.append(val2)
        return feature_vector

    # TODO manually make some features more/less important
    # report importance of each individual feature in clustering
    # try seeding with some gestures? -- bias, but since we're doing it based on prior gesture research it's ok

    # TODO check what audio features are in original paper

    # Audio for agent is bad -- TTS is garbaggio
    # assumes there is good prosody in voice (TTS there isn't)

    # Co-optimize gesture-language clustering (learn how)
    # KL distance for two clusters?

    # learn similarity of sentences from within one gesture
    # how to map gesture clusters <--> sentence clusters
    # in the end want to optimize overlapping clusters btw gesture/language

    # probabilistic mapping of sentence (from gesture) to sentence cluster

    ##############################################################
    # Helpers/Calculators ########################################
    ##############################################################
    @timeit
    def _get_rl_hand_keypoints(self, gesture, hand):
        keys = []
        keypoint_range = ALL_RIGHT_HAND_KEYPOINTS if hand == 'r' else ALL_LEFT_HAND_KEYPOINTS
        if not gesture['keyframes']:
            print("No keyframes found for gesture")
            print(gesture)
            return

        for t in gesture['keyframes']:
            if not isinstance(t, dict):
                print("found empty keyframes for gesture %s" % gesture['id'])
                print(gesture)
                # TODO fix this, known temporary fix
                return [{'y': [0], 'x': [0]}]
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
        return np.linalg.norm(feat1 - feat2)

    @timeit
    def _calculate_distance_between_vectors(self, v1, v2):
        return np.linalg.norm(np.array(v1) - np.array(v2))

    def _log(self, s):
        self.logs.append(s)

    def _write_logs(self):
        with open(self.logfile, 'w') as f:
            for log in self.logs:
                f.write("%s\n" % log)
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
                min_d = dist
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

    # a random helper for me to find centroids
    # and peep average distances
    def _calc_dist_between_random_gestures(self):
        i = random.randrange(0, len(self.agd))
        j = random.randrange(0, len(self.agd))
        return self._calculate_distance_between_vectors(self.agd[i]['feature_vec'], self.agd[j]['feature_vec'])

    # lol this is wrong.
    # goes through each gesture twice (which actually might calculate right value but is wrong theoretically)
    def get_avg_within_cluster_distances(self, cluster_id):
        c = self.clusters[cluster_id]
        gs = c['gestures']
        all_dists = []
        for i in range(len(gs)):
            for j in range(len(gs)):
                if i == j:
                    continue
                all_dists = all_dists + self._calculate_distance_between_vectors(gs[i]['feature_vec'],
                                                                                 gs[j]['feature_vec'])

        dists = {'average': _avg(all_dists), 'max': max(all_dists), 'min': min(all_dists)}
        return dists

    # takes a cluster ID, returns nearest neighbor cluster ID
    def get_nearest_cluster_id(self, cluster_id):
        dist = 1000
        min_c = 0
        for c in self.clusters:
            if c == cluster_id:
                continue
            mind = self._calculate_distance_between_vectors(self.clusters[c]['centroid'],
                                                            self.clusters[cluster_id]['centroid'])
            if mind < dist:
                dist = mind
                min_c = c
        return min_c

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
        return _avg(dists)

    # gets silhouette score for cluster using centroid
    def get_silhouette_score(self, cluster_id):
        p = self.clusters[cluster_id]['centroid']
        a = self.get_avg_dist_between_point_and_cluster(p, cluster_id)
        b = self.get_avg_dist_between_point_and_cluster(p, self.get_nearest_cluster_id(cluster_id))
        score = (b - a) / max(b, a)
        return score

    def get_avg_silhouette_score(self):
        scores = []
        for g in self.clusters:
            scores.append(self.get_silhouette_score(g))
        return _avg(scores)

# our basic problem is that we need to figure out how to map distances between motions that are very long
# vectors, and different lengths of keyframes. But we need to distinguish between the speed of those motions
# as well...

# Another big issue is that individuals get clustered together BECAUSE their large-scale movements are
# similar. But this might not be so much of an issue... if we find patterns that are common to a movement pattern,
# then it's just a case of a gesture cluster representing a personality that expresses a particular trait.


# Silhouette scores for clusters are a good way of determining how many "base" gestures there may be??
