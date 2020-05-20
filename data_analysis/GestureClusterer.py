#!/usr/bin/env pythons
import json
import os
import numpy as np
import random
import pandas as pd
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


# clusters look like this:
# {'gesture_ids': [# list of ids #], 'centroid': [# feature vector #]}


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


@timeit
def _get_rl_hand_keypoints(gesture, hand):
    keys = []
    keypoint_range = ALL_RIGHT_HAND_KEYPOINTS if hand == 'r' else ALL_LEFT_HAND_KEYPOINTS
    if not gesture['keyframes']:
        print("No keyframes found for gesture")
        print(gesture)
        return

    for t in gesture['keyframes']:
        if not isinstance(t, dict):
            # print("found empty keyframes for gesture %s" % gesture['id'])
            # TODO fix this, known temporary fix
            return [{'y': [0], 'x': [0]}]
        y = [t['y'][i] for i in keypoint_range]
        x = [t['x'][i] for i in keypoint_range]
        keys.append({'y': y, 'x': x})
    return keys


# Some special alternative clusterings
def create_max_difference_matrix_max_different_frame(df):
    order = list(zip(df.id, df.keyframes))  # keep dict in order to sort and
    ordered_keys = []
    for k, v in sorted(order, key=sort_indexes):  # assign proper distances to it.
        ordered_keys.append(v)
    similarities = []
    for i in tqdm(range(len(ordered_keys))):
        keys = ordered_keys[i]
        comparison_frame = get_max_different_frame_in_gesture(keys)
        similarities.append(
            [get_frame_diff(keys[comparison_frame], k2[get_max_different_frame_in_gesture(k2)]) for k2 in ordered_keys])
    return similarities


def cluster_gestures_by_max_different_frame(df):
    similarities = create_max_difference_matrix_max_different_frame(df)
    ac = AgglomerativeClustering(n_clusters=10, affinity='precomputed', linkage='complete')
    u = ac.fit_predict(similarities)

    order = list(zip(df.id, df.keyframes))
    clusters = {}
    for i in range(len(order)):
        if u[i] not in clusters.keys():
            clusters[u[i]] = {'gestures': []}
    clusters[u[i]]['gestures'].append(order[i][0])
    return clusters


def get_percent_gesture_cluster_overlap(A, B):
    setsA = [(c, set(A[c]['gesture_ids'])) for c in A.keys()]
    setsB = [(c, set(B[c]['gesture_ids'])) for c in B.keys()]

    overlaps = {}
    for k, a_ids in setsA:
        max_overlap_id = 0
        max_overlap = 0
        all_overlaps = []
        for j, b_ids in setsB:
            overlap = float(len(a_ids & b_ids)) / len(a_ids)
            print("overlap for ", k, j, ":", overlap)
            all_overlaps.append(overlap)
            if overlap > max_overlap:
                max_overlap = overlap
                max_overlap_id = j
        overlaps[k] = {
            'max_overlap': max_overlap,
            'max_overlap_id': max_overlap_id,
            'avg_overlap': np.array(all_overlaps).mean(),
            'std_overlap': np.array(all_overlaps).std()
        }
    return overlaps


def get_avg_motion_dist_for_clusters(df, clusters, motion_metric='feature_vec'):
    avg = np.array([])
    for c in clusters.keys():
        fvs = df[df['id'].isin(clusters[c]['gesture_ids'])]['motion_feature_vec'].tolist()
        dists = []
        for i in range(len(fvs)):     # maybe inefficient to get all the gestures one by one?
            if not i % 200:
                print(i, "/", len(fvs))
            for j in range(i, len(fvs)):
                diff = np.linalg.norm(np.array(fvs[i]) - np.array(fvs[j]))
                dists.append(diff)
        avg = np.append(avgs, np.array(dists).mean())
    return avg


def _calculate_distance_between_vectors(v1, v2):
    return np.linalg.norm(np.array(v1) - np.array(v2))


class GestureClusterer:
    # all the gesture data for gestures we want to cluster.
    # the ids of any seed gestures we want to use for our clusters.
    def __init__(self, gesture_df):
        # I have no idea what best practices are but I'm almost certain this is
        # a gross, disgusting anti-pattern for iterating IDs.
        self.c_id = 0
        self.df = gesture_df
        self.clusters = {}
        self.clf = NearestCentroid()
        # todo make this variable
        homePath = os.getenv("HOME")
        self.cluster_file = os.path.join(homePath, "GestureData", "cluster_tmp.json")
        self.has_assigned_feature_vecs = False
        self.total_clusters_created = 0
        tqdm.pandas()

    def clear_clusters(self):
        self.clusters = {}
        self.total_clusters_created = 0
        self.c_id = 0

    @timeit
    def cluster_gestures(self, gesture_features=GESTURE_FEATURES, max_cluster_distance=0.03,
                         max_number_clusters=0, seed_ids=None):
        if seed_ids is None:
            seed_ids = []

        if 'motion_feature_vec' in list(self.df):
            print("already have feature vectors in our gesture data")
        else:
            print("trying to assign feature vectors")
            self.df = self.assign_feature_vectors(gesture_features)

        # if we're seeding our clusters with specific gestures
        if len(seed_ids):
            for s_id in seed_ids:
                row_id = self.df.index[self.df['id'] == s_id].tolist()
                if not row_id:
                    print("could not locate gesture ", s_id)
                fv = self.df.iloc[row_id]['motion_feature_vec']
                self._create_new_cluster(s_id, fv)

        self._cluster_gestures(max_cluster_distance, max_number_clusters, seed_ids)

    @timeit
    def _cluster_gestures(self, max_cluster_distance=0.03, max_number_clusters=0, seed_ids=None):
        if seed_ids is None:
            seed_ids = []
        print("Clustering gestures")

        self.df.progress_apply(
            lambda row: self._cluster_gesture_from_row(row, max_cluster_distance, max_number_clusters, seed_ids),
            axis=1)
        # now recluster based on where the new centroids are
        self._recluster_by_centroids()
        print("created %s clusters" % self.total_clusters_created)
        return

    def _cluster_gesture_from_row(self, gesture, max_cluster_distance=0.03, max_number_clusters=0, seed_ids=None):
        if gesture['id'] in seed_ids:
            return
        (nearest_cluster_id, nearest_cluster_dist) = self._get_shortest_cluster_dist(gesture['motion_feature_vec'])
        if max_number_clusters and len(self.clusters) > max_number_clusters:
            self._add_gesture_to_cluster(gesture['id'], nearest_cluster_id)
        # we're further away than we're allowed to be, OR this is the first cluster.
        elif (max_cluster_distance and nearest_cluster_dist > max_cluster_distance) or (not len(self.clusters)):
            self._create_new_cluster(gesture['id'], gesture['motion_feature_vec'])
        else:
            self._add_gesture_to_cluster(gesture['id'], nearest_cluster_id)

    @timeit
    def _add_gesture_to_cluster(self, gesture_id, cluster_id):
        try:
            self.clusters[cluster_id]['gesture_ids'].append(gesture_id)
            self._update_cluster_centroid(cluster_id)
        except RuntimeError as e:
            print('could not add gesture %s to cluster %s' % (gesture_id, cluster_id))
            print('cluster keys:')
            print(e)
            print(self.clusters[cluster_id].keys())

    @timeit
    def assign_feature_vectors(self, gesture_features=GESTURE_FEATURES):
        df = self.df
        print("Getting initial feature vectors.")
        # TODO track this through and make sure it's assigning the right thing to the right thing
        feats = list(
            df.progress_apply(lambda row: self._get_gesture_features(row, gesture_features=gesture_features), axis=1))
        normalized_feats = list(self._normalize_feature_values(feats))
        df['motion_feature_vec'] = normalized_feats
        return df

    def get_feature_vector_by_gesture_id(self, g_id):
        i = self.df.index[self.df['id'] == g_id].tolist()
        if i:
            return self.df.iloc[i[0]]['motion_feature_vec']
        return

    @timeit
    def _normalize_feature_values(self, feat_vecs):
        print("Normalizing feature vectors.")
        feat_vecs = np.array(feat_vecs)
        feat_vecs_normalized = _normalize_across_features(feat_vecs)
        return feat_vecs_normalized

    @timeit
    def _create_new_cluster(self, seed_gesture_id, feature_vec):
        new_cluster_id = self.c_id
        self.c_id = self.c_id + 1
        c = {
            'cluster_id': new_cluster_id,
            'centroid': feature_vec,
            'seed_id': seed_gesture_id,
            'gesture_ids': [seed_gesture_id]
        }
        self.clusters[new_cluster_id] = c
        self.total_clusters_created += 1

    # now that we've done the clustering, recluster and only allow clusters to form around current centroids.
    # TODO go through and recluster any clusters whose silhouette scores are negative?
    def _recluster_by_centroids(self):
        print("Reclustering by centroid")
        # clear old gestures
        for c in self.clusters:
            self.clusters[c]['gesture_ids'] = []
        self.df.progress_apply(self._just_assign_cluster, axis=1)
        return

    def _just_assign_cluster(self, gesture):
        min_c, dist = self._get_shortest_cluster_dist(gesture['motion_feature_vec'])
        self.clusters[min_c]['gesture_ids'].append(gesture['id'])

    def get_sentences_by_cluster(self, cluster_id):
        c = self.clusters[cluster_id]
        gestures_in_cluster = self.df.loc[self.df['id'].isin(c['gesture_ids'])]
        transcripts = list(gestures_in_cluster['transcripts'])
        return transcripts

    def get_gesture_ids_by_cluster(self, cluster_id):
        c = self.clusters[cluster_id]
        return c['gesture_ids']

    def report_clusters(self):
        print(("Number of clusters: %s" % len(self.clusters)))
        cluster_rep = [(c, len(self.clusters[c]['gesture_ids'])) for c in list(self.clusters.keys())]
        cluster_lengths = [len(self.clusters[c]['gesture_ids']) for c in list(self.clusters.keys())]
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
        gestures_in_cluster = self.df.loc[self.df['id'].isin(c['gesture_ids'])]
        feat_vecs = list(gestures_in_cluster['motion_feature_vec'])
        dists = [_calculate_distance_between_vectors(f, cent) for f in feat_vecs]
        return np.average(dists)

    # instead of this need to use centroid.
    @timeit
    def _get_shortest_cluster_dist(self, feature_vec):
        shortest_dist = 10000
        nearest_cluster_id = ''
        for k in self.clusters:
            c = self.clusters[k]
            centroid = c['centroid']
            dist = _calculate_distance_between_vectors(feature_vec, centroid)
            if dist < shortest_dist:
                nearest_cluster_id = c['cluster_id']
            shortest_dist = min(shortest_dist, dist)
        return nearest_cluster_id, shortest_dist

    def _update_cluster_centroid(self, cluster_id):
        c = self.clusters[cluster_id]
        gestures_in_cluster = self.df.loc[self.df['id'].isin(c['gesture_ids'])]
        feat_vecs = list(gestures_in_cluster['motion_feature_vec'])
        feat_vecs = np.array(feat_vecs)
        c['centroid'] = [np.average(x) for x in feat_vecs.T]
        self.clusters[cluster_id] = c
        return

    ############################################################
    # MOVEMENT CHARACTERISTICS #################################
    ############################################################
    @timeit
    def _get_gesture_features(self, gesture, gesture_features=GESTURE_FEATURES):
        r_keyframes = _get_rl_hand_keypoints(gesture, 'r')
        l_keyframes = _get_rl_hand_keypoints(gesture, 'l')
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
    def _calculate_distance_between_gestures(self, g1, g2):
        if 'feature_vec' in list(g1.keys()) and 'feature_vec' in list(g2.keys()):
            return np.linalg.norm(np.array(g1['feature_vec']) - np.array(g2['feature_vec']))

        feat1 = np.array(self._get_gesture_features(g1))
        feat2 = np.array(self._get_gesture_features(g2))
        return np.linalg.norm(feat1 - feat2)

    def get_closest_gesture_to_centroid(self, cluster_id):
        c = self.clusters[cluster_id]
        cent = c['centroid']
        min_d = 1000
        g_id = 0
        for g in c['gestures']:
            dist = _calculate_distance_between_vectors(g['feature_vec'], cent)
            if dist < min_d:
                g_id = g['id']
                min_d = dist
        return g_id

    def get_random_gesture_id_from_cluster(self, cluster_id):
        c = self.clusters[cluster_id]
        i = random.randrange(0, len(c['gestures']))
        return c['gestures'][i]['id']

    # a random helper for me to find centroids
    # and peep average distances
    def _calc_dist_between_random_gestures(self):
        i = random.randrange(0, len(self.agd))
        j = random.randrange(0, len(self.agd))
        return _calculate_distance_between_vectors(self.agd[i]['feature_vec'], self.agd[j]['feature_vec'])

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
                all_dists = all_dists + _calculate_distance_between_vectors(gs[i]['feature_vec'],
                                                                                 gs[j]['feature_vec'])

        dists = {'average': _avg(all_dists), 'max': max(all_dists), 'min': min(all_dists)}
        return dists

    # takes a cluster ID, returns nearest neighbor cluster ID
    def get_nearest_cluster_from_cluster_id(self, cluster_id):
        dist = 1000
        min_c = 0
        for c in self.clusters:
            if c == cluster_id:
                continue
            mind = _calculate_distance_between_vectors(self.clusters[c]['centroid'],
                                                            self.clusters[cluster_id]['centroid'])
            if mind < dist:
                dist = mind
                min_c = c
        return min_c

    # given a point and cluster id, returns avg distance between the point and
    # all points in the cluster.
    def get_dists_between_point_and_cluster(self, vec, cluster_id, clusters=None):
        if not clusters:
            clusters = self.clusters
        dists = []
        c = clusters[cluster_id]
        if len(c['gesture_ids']) == 0:
            print("WARNING: NO GESTURES FOUND IN CLUSTER ID %s" % cluster_id)
            print("num clusters: %s" % len(clusters))
            return 0
        for g_id in c['gesture_ids']:
            f = self.get_feature_vector_by_gesture_id(g_id)
            dists.append(_calculate_distance_between_vectors(vec, f))
        return np.array(dists)

    # gets silhouette score for cluster using centroid
    def get_silhouette_score(self, cluster_id):
        c = self.clusters[cluster_id]
        c2 = self.get_nearest_cluster_from_cluster_id(cluster_id)
        c_magnitude = len(c['gesture_ids'])
        c2_magnitude = len(self.clusters[c2]['gesture_ids'])
        if len(c['gesture_ids']) == 1:
            return 0
        elif len(c['gesture_ids'] < 1):
            print("No gestures in cluster ", cluster_id)
            return 0
        p = c['centroid']
        a = sum(self.get_dists_between_point_and_cluster(p, cluster_id)) / (c_magnitude - 1)
        b = sum(self.get_dists_between_point_and_cluster(p, c2)) / c2_magnitude
        score = (b - a) / max(b, a)
        return score

    # get silhouette score of MOVEMENT as if it were clustered as clusters.
    def get_silhouette_score_for_alternative_clustering(self, clusters, cluster_id):
        c = clusters[cluster_id]
        cluster_magnitude = len(c['gesture_ids'])
        if len(c['gesture_ids']) <= 1:
            return 0
        p = self.get_feature_vector_by_gesture_id(c['centroid_id'])
        a = sum(self.get_dists_between_point_and_cluster(p, cluster_id, clusters=clusters)) / (cluster_magnitude - 1)
        b = sum(self.get_dists_between_point_and_cluster(p,
                                                         self.get_nearest_cluster_from_cluster_id(cluster_id),
                                                         clusters=clusters) /
                cluster_magnitude)
        score = (b - a) / max(b, a)
        return score

    def get_avg_silhouette_score(self):
        scores = []
        for g in self.clusters:
            scores.append(self.get_silhouette_score(g))
        return _avg(scores)

    # given two sets of clusterings A and B, determines what percentage of gestures in each
    # cluster in clustering A are also in the same cluster in clustering B

# our basic problem is that we need to figure out how to map distances between motions that are very long
# vectors, and different lengths of keyframes. But we need to distinguish between the speed of those motions
# as well...

# Another big issue is that individuals get clustered together BECAUSE their large-scale movements are
# similar. But this might not be so much of an issue... if we find patterns that are common to a movement pattern,
# then it's just a case of a gesture cluster representing a personality that expresses a particular trait.


# Silhouette scores for clusters are a good way of determining how many "base" gestures there may be??
