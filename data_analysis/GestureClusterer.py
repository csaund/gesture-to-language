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
from sklearn.cluster import AgglomerativeClustering, DBSCAN


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

# if a cluster has only 1 gesture, re-cluster into others using supplied distance metric
def combine_singletons(clusters, distance_metric='feature_vec'):
    to_del = []
    for k in clusters.keys():
        if len(clusters[k]['gesture_ids']) == 0:
            to_del.append(k)
            continue
        if len(clusters[k]['gesture_ids']) == 1:
            to_del.append(k)
            nearest_k = get_nearest_cluster_from_cluster_id(k, clusters)
            if 'combined_keys' not in clusters[nearest_k].keys():
                clusters[nearest_k]['combined_keys'] = [nearest_k, k]
            else:
                clusters[nearest_k]['combined_keys'].append(k)
            clusters[nearest_k]['gesture_ids'] += clusters[k]['gesture_ids']

    for k in to_del:
        del clusters[k]
    return clusters


def combine_clusters_by_id(cid1, cid2, clustering, df):
    c1 = clustering[cid1]
    c2 = clustering[cid2]
    nk = str(cid1) + '+' + str(cid2)
    ng_ids = c1['gesture_ids'] + c2['gesture_ids']
    comb_keys = []
    if 'combined_keys' in c1.keys() and 'combined_keys' in c2.keys():
        comb_keys = c1['combined_keys'] + c2['combined_keys']
    elif 'combined_keys' in c1.keys():
        comb_keys = c1['combined_keys']
    elif 'combined_keys' in c2.keys():
        comb_keys = c2['combined_keys']
    clustering[nk] = {
        'gesture_ids': ng_ids,
        'combined_keys': comb_keys,
        'centroid': get_centroid_from_gids(df, ng_ids)
    }
    del clustering[cid1]
    del clustering[cid2]
    clustering[nk]['silhouette_score'] = \
        get_silhouette_score_for_alternative_clustering_by_id(df, clustering, nk)
    return clustering


def get_key_for_worst_silhouette_score(clusters):
    min_score = 1
    mink = 0
    for k in clusters.keys():
        c = clusters[k]
        s = c['silhouette_score']
        if s < min_score:
            min_score = s
            mink = k
    return mink


def drop_worst_n_clusters(clusters, n):
    cs = clusters
    for i in range(n):
        k = get_key_for_worst_silhouette_score(cs)
        del cs[k]
    return cs


def combine_worst_n_clusters(clusters, n, df):
    for i in tqdm(range(n)):
        k = get_key_for_worst_silhouette_score(clusters)
        ck = get_nearest_cluster_from_cluster_id(k, clusters)
        clusters = combine_clusters_by_id(k, ck, clusters, df)
    return clusters


def get_silhouette_scores_alternative_clustering(df, clusters, exclude_keys=[], add_scores=True):
    scores = []
    for c in tqdm(clusters.keys()):
        if c in exclude_keys:
            continue
        elif 'silhouette_score' in clusters[c].keys():
            scores.append(clusters[c]['silhouette_score'])
            continue
        s = get_silhouette_score_for_alternative_clustering_by_id(df, clusters, c)
        scores.append(s)
        if add_scores:
            clusters[c]['silhouette_score'] = s
    if add_scores:
        return np.array(scores), clusters
    return np.array(scores)


# TODO do somethng with this
def get_speaker_distribution(df, clusters):
    dists = []
    sps = df.speaker.unique().tolist()
    for c in tqdm(clusters.keys()):
        dist = {}
        for s in sps:
            dist[s] = len(df[(df['id'].isin(clusters[c]['gesture_ids'])) & (df['speaker'] == s)])
        # speaker_dist = get_speaker_distribution_for_ids(df, clusters[c]['gesture_ids'])
        dists.append(dist)
    # TODO implement Ripley L/K functions here
    return dists


def get_speaker_distribution_for_ids(df, ids):
    sps = df.speaker.unique().tolist()
    speakers = dict(zip(sps, [0]*len(sps)))
    for i in ids:
        sp = df[df['id'] == i]['speaker'].to_list()[0]
        speakers[sp] += 1
    return speakers



def len_vs_sil_score_scatter(clusters):
    lens = [len(clusters[c]['gesture_ids']) for c in clusters.keys()]
    sils = [clusters[c]['silhouette_score'] for c in clusters.keys()]
    plt.scatter(lens, sils)


def get_silhouette_score_for_alternative_clustering_by_id(df, clusters, cluster_id):
    c = clusters[cluster_id]
    cluster_magnitude = len(c['gesture_ids'])
    if len(c['gesture_ids']) <= 1:
        return 0
    p = c['centroid']
    a = sum(get_dists_between_point_and_cluster(df, p, cluster_id, clusters=clusters)) / (cluster_magnitude - 1)
    b = sum(get_dists_between_point_and_cluster(df, p,
                                                     get_nearest_cluster_from_cluster_id(cluster_id, clusters=clusters),
                                                     clusters=clusters)
                                                     / cluster_magnitude)
    score = (b - a) / max(b, a)
    return score


def normalize_feature_vectors_themselves(df):
    df['motion_feature_vec'] = df['motion_feature_vec'].progress_apply(lambda x: normalize_to(x, 1))
    return df


def normalize_to(v, n):
    return v / (v.max()/n)


def get_between_cluster_distances(clusters):
    ds = []
    for c in clusters.keys():
        nc = get_nearest_cluster_from_cluster_id(c, clusters=clusters)
        dist = _calculate_distance_between_vectors(clusters[c]['centroid'], clusters[nc]['centroid'])
        ds.append(dist)
    return np.array(ds)


# TODO MOVE THIS
def no_singletons(clusters):
    new_clusters = {}
    for c in clusters.keys():
        if len(clusters[c]['gesture_ids']) <= 1:
            continue
        new_clusters[c] = clusters[c]
    return new_clusters


def get_feature_vector_by_gesture_id(df, g_id):
    g = df[df['id'] == g_id]
    if not len(g):
        print("could not find gesture ", g_id)
        return
    return g['motion_feature_vec'].to_list()[0]


# clusters look like this:
# {'gesture_ids': [# list of ids #], 'centroid': [# feature vector #]}
def add_centroids_to_clusters_motion_vec(df, clusters):
    for c in tqdm(clusters.keys()):
        clusters[c]['centroid'] = get_centroid_from_gids(df, clusters[c]['gesture_ids'])
    return clusters


def get_centroid_from_gids(df, gids):
    fvs = []
    for gid in gids:
        fv = get_feature_vector_by_gesture_id(df, gid)
        fvs.append(fv)
    m = np.array(fvs)
    return m.mean(0)


def get_dists_between_point_and_cluster(df, vec, cluster_id, clusters):
    dists = []
    c = clusters[cluster_id]
    if len(c['gesture_ids']) == 0:
        print("WARNING: NO GESTURES FOUND IN CLUSTER ID %s" % cluster_id)
        print("num clusters: %s" % len(clusters))
        return 0
    for g_id in c['gesture_ids']:
        f = get_feature_vector_by_gesture_id(df, g_id)
        dists.append(_calculate_distance_between_vectors(vec, f))
    return np.array(dists)


def get_nearest_cluster_from_cluster_id(cluster_id, clusters=None):
    dist = 10000
    min_c = cluster_id
    for c in clusters.keys():
        if c == cluster_id:
            continue
        mind = _calculate_distance_between_vectors(clusters[c]['centroid'],
                                                   clusters[cluster_id]['centroid'])
        if mind < dist:
            dist = mind
            min_c = c
    return min_c


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


# TODO turn this into list comprehension
# Some special alternative clusterings
def create_max_difference_matrix_max_different_frame(df):
    order = list(zip(df.id, df.keyframes))  # keep dict in order to sort and
    ordered_keys = []
    max_diff_frame = []
    for k, v in tqdm(sorted(order, key=sort_indexes), position=0):  # assign proper distances to it.
        if not v or isinstance(v, dict):
            ordered_keys.append(None)
            max_diff_frame.append(None)
            continue
        ordered_keys.append(v)
        max_diff_frame.append(get_max_different_frame_in_gesture(v))
    similarities = []
    for i in tqdm(range(len(ordered_keys)), position=0):
        for j in range(len(ordered_keys)):
            if not ordered_keys[i] or not ordered_keys[j]:
                similarities.append(None)
                continue
            similarities.append(get_frame_diff(ordered_keys[i][max_diff_frame[i]], ordered_keys[j][max_diff_frame[j]]))
    return similarities


def cluster_gestures_by_max_different_frame(df, n_clusters=10, algorithm='agglomerative'):
    if not algorithm:
        algorithm = 'agglomerative'
    similarities = create_max_difference_matrix_max_different_frame(df)
    clustering = None
    if algorithm == 'agglomerative':
        clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='complete')
    elif algorithm == 'dbscan':
        clustering = DBSCAN(metric='precomputed')
    else:
        print("unrecognized algorithm", algorithm)
        print("please choose one of: agglomerative, dbscan")

    u = clustering.fit_predict(similarities)

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
        avg = np.append(avg, np.array(dists).mean())
    return avg


# can be used as a metric to see how well clusterings matched
def get_gesture_motion_distances(df, motion_metric='feature_vec'):
    fvs = df['motion_feature_vec'].tolist()
    dists = []
    for i in range(len(fvs)):  # maybe inefficient to get all the gestures one by one?
        if not i % 200:
            print(i, "/", len(fvs))
        for j in range(i, len(fvs)):
            diff = np.linalg.norm(np.array(fvs[i]) - np.array(fvs[j]))
            dists.append(diff)
    return np.array(dists)

# avg_rhet_unit_dbscan_motion_dist = get_avg_motion_distance_for_clusters(df, rhet_clust_unit)
# total_dists = get_gesture_motion_distances
# plt.scatter(y=avg_rhet_unit_dbscan_motion_dist / total_dists.mean(), x=[len(rhet_clust_unit[c]['gesture_ids']) for c in rhet_clust_unit.keys()])
# voila, a plot that shows they aren't that great, cause it should be near 0


def _calculate_distance_between_vectors(v1, v2):
    return np.linalg.norm(np.array(v1) - np.array(v2))


# Given these ids, perform k-means on these IDs.
# (create multiple clusters from these IDs, based on motion)
# def create_subclusters_from_gesture_ids():



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
    def assign_feature_vectors(self, df=None, gesture_features=GESTURE_FEATURES, inplace=True):
        if df is None:
            df = self.df
        print("Getting initial feature vectors.")
        # TODO track this through and make sure it's assigning the right thing to the right thing
        feats = list(
            df.progress_apply(lambda row: self._get_gesture_features(row, gesture_features=gesture_features), axis=1))
        normalized_feats = list(self._normalize_feature_values(feats))
        df['motion_feature_vec'] = normalized_feats
        if inplace:
            self.df = df
        return df

    def get_feature_vector_by_gesture_id(self, g_id):
        try:
            return self.df[self.df['id'] == g_id].motion_feature_vec.to_list()[0]
        except:
            print('no id found ', gid)
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
        return self.df[self.df['id'].isin(c['gesture_ids'])]['transcript'].to_list()

    def get_gesture_ids_by_cluster(self, cluster_id):
        c = self.clusters[cluster_id]
        return c['gesture_ids']

    def report_clusters(self):
        report = {}
        report["NumberOfClusters"] = len(self.clusters)
        cluster_rep = [(c, len(self.clusters[c]['gesture_ids'])) for c in list(self.clusters.keys())]
        cluster_lengths = [len(self.clusters[c]['gesture_ids']) for c in list(self.clusters.keys())]
        report["ClusterLengths"] = cluster_rep
        report["AvgClusterSize" ] = np.average(cluster_lengths)
        report["MedianClusterSize"] = np.median(cluster_lengths)
        report["LargestClusterSize"] = max(cluster_lengths)
        cluster_sparsity = [self.get_cluster_sparsity(c) for c in list(self.clusters.keys())]
        report["ClusterSparsity" ] = cluster_sparsity
        report["AvgClusterSparsity" ] = np.average(cluster_sparsity)
        report["MedianClusterSparsity"] = np.median(cluster_sparsity)
        report["AvgSilhouetteScore"] = self.get_avg_silhouette_score()
        sil_scores_keys = [(c, self.clusters[c]['silhouette_score']) for c in self.clusters.keys()]
        report['AllSilhouetteScores'] = sil_scores_keys
        # TODO: average and median centroid distances from each other.
        # TODO: also get minimum and maximum centroid distances.
        return report

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
        for g in c['gesture_ids']:
            dist = _calculate_distance_between_vectors(g['feature_vec'], cent)
            if dist < min_d:
                g_id = g['id']
                min_d = dist
        return g_id

    def get_random_gesture_id_from_cluster(self, cluster_id):
        c = self.clusters[cluster_id]
        i = random.randrange(0, len(c['gesture_ids']))
        return c['gesture_ids'][i]['id']

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
    def get_nearest_cluster_from_cluster_id(self, cluster_id, clusters=None):
        if clusters is None:
            clusters = self.clusters
        dist = 1000
        min_c = 0
        for c in clusters.keys():
            if c == cluster_id:
                continue
            mind = _calculate_distance_between_vectors(clusters[c]['centroid'],
                                                       clusters[cluster_id]['centroid'])
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
        elif len(c['gesture_ids']) < 1:
            print("No gestures in cluster ", cluster_id)
            return 0
        p = c['centroid']
        a = sum(self.get_dists_between_point_and_cluster(p, cluster_id)) / (c_magnitude - 1)
        b = sum(self.get_dists_between_point_and_cluster(p, c2)) / c2_magnitude
        score = (b - a) / max(b, a)
        self.clusters[cluster_id]['silhouette_score'] = score
        return score

    # get silhouette score of MOVEMENT as if it were clustered as clusters.
    def get_silhouette_score_for_alternative_clustering(self, clusters, cluster_id):
        c = clusters[cluster_id]
        cluster_magnitude = len(c['gesture_ids'])
        if len(c['gesture_ids']) <= 1:
            return 0
        scores = []
        for gid in c['gesture_ids']:
            p = self.get_feature_vector_by_gesture_id(gid)
            a = sum(self.get_dists_between_point_and_cluster(p, cluster_id, clusters=clusters)) / (cluster_magnitude - 1)
            b = sum(self.get_dists_between_point_and_cluster(p,
                                                             self.get_nearest_cluster_from_cluster_id(cluster_id),
                                                             clusters=clusters) /
                    cluster_magnitude)
            score = (b - a) / max(b, a)
            scores.append(score)
        scores = np.array(scores)
        return scores.mean()

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
def report_clusters(GC):
    report = {}
    report["NumberOfClusters"] = len(GC.clusters)
    cluster_rep = [(c, len(GC.clusters[c]['gesture_ids'])) for c in list(GC.clusters.keys())]
    cluster_lengths = [len(GC.clusters[c]['gesture_ids']) for c in list(GC.clusters.keys())]
    report["ClusterLengths"] = cluster_rep
    report["AvgClusterSize" ] = np.average(cluster_lengths)
    report["MedianClusterSize"] = np.median(cluster_lengths)
    report["LargestClusterSize"] = max(cluster_lengths)
    cluster_sparsity = [GC.get_cluster_sparsity(c) for c in list(GC.clusters.keys())]
    report["ClusterSparsity" ] = cluster_sparsity
    report["AvgClusterSparsity" ] = np.average(cluster_sparsity)
    report["MedianClusterSparsity"] = np.median(cluster_sparsity)
    try:
        report["AvgSilhouetteScore"] = GC.get_avg_silhouette_score()
        sil_scores_keys = [(c, GC.clusters[c]['silhouette_score']) for c in GC.clusters.keys()]
        report['AllSilhouetteScores'] = sil_scores_keys
        # TODO: average and median centroid distances from each other.
        # TODO: also get minimum and maximum centroid distances.
    except:
        print("Could not get silhouette score for clusters")
    return report
