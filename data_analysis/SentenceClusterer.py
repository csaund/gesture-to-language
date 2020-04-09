print("loading tf")
# import tensorflow as tf
print("loading tfh")
# import tensorflow_hub as hub
import random
import numpy as np
from tqdm import tqdm
from common_helpers import *
from termcolor import colored

import time

print("loading nltk")
## TODO: use this?
import nltk
from nltk import sentiment as sent
from nltk import word_tokenize, pos_tag
from nltk.corpus import wordnet as wn


# tf.compat.v1.disable_eager_execution()

MAX_CLUSTER_SIZE = 90
CLUSTER_SIMILARITY_TOLERANCE = 0.6
MAX_NUMBER_CLUSTERS = 10000

# from NewSentenceClusterer import *
# SC = SentenceClusterer("rock")
# SC.cluster_sentences(gd)
# if you previously saved a run of agd from this, so you don't need to get sentence embeddings again.

class SentenceClusterer():
    def __init__(self, speaker, seeds=[]):
        self.speaker = speaker
        print("oh boy we're in for it now.")
        # self.embed_fn = self.initialize_encoder("https://tfhub.dev/google/universal-sentence-encoder/2")
        self.has_assigned_feature_vecs = False
        self.logfile = "%s/gesture-to-language/sentence_cluster_logs.txt" % os.getenv("HOME")
        self.full_transcript_bucket = "full_timings_with_transcript_bucket"
        self.clusters = {}
        self.logs = []
        self.agd = None
        self.get_transcript()
        self.c_id = 0
        # TODO implement seed gestures

    def initialize_encoder(self, module):
        with tf.compat.v1.Graph().as_default():
            sentences = tf.compat.v1.placeholder(tf.string)
            embed = hub.Module(module)
            embeddings = embed(sentences)
            session = tf.compat.v1.train.MonitoredSession()
        return lambda x: session.run(embeddings, {sentences: x})

    def get_transcript(self):
        if self.agd:
            return
        fp = "temp.json"
        download_blob(self.full_transcript_bucket,
                      "%s_timings_with_transcript.json" % self.speaker,
                      fp)
        self.agd = read_data(fp)
        os.remove("temp.json")

    def encode_sentence(self, item):
        return self.embed_fn(item)

    # takes two strings
    def get_sentence_embedding(self, s):
        return self.embed_fn([s])

    def get_paragraph_embedding(self, p):
        return self.embed_fn(p)

    # takes two strings
    def get_sentence_similarity(self, s1, s2):
        s1_mat = self.embed_fn([s1])
        s2_mat = self.embed_fn([s2])
        return np.inner(s1_mat, s2_mat).max()

    def get_gesture_sentence_similarity(self, g1, g2):
        g1_mat = [g1['sentence_embedding']]
        g2_mat = [g2['sentence_embedding']]
        return np.inner(g1_mat, g2_mat).max()

    def get_sentence_similarity_to_paragraph(self, s, p):
        s_mat = self.embed_fn([s])
        p_mat = self.embed_fn(p)
        return np.inner(s_mat, p_mat).max()

    def get_embedded_sentence_similarity_to_paragraph(self, s_mat, p):
        p_mat = self.embed_fn(p)
        return np.inner(s_mat, p_mat).max()

    def get_embedded_sentence_similarity_to_cluster(self, s_mat, cluster_id):
        p_mat = self.clusters[cluster_id]['cluster_embedding']
        return np.inner(s_mat, p_mat).max()

    def get_distance_between_clusters(self, c_id1, c_id2):
        c1 = self.clusters[c_id1]['cluster_embedding']
        c2 = self.clusters[c_id2]['cluster_embedding']
        return np.inner(c1, c2).max()

    def get_distance_between_sentence_groups(self, sents1, sents2):
        c1 = self.embed_fn(sents1)
        c2 = self.embed_fn(sents2)
        return np.inner(c1, c2).max()

    def filter_agd(self, exclude_ids):
        agd = {'phrases': [d for d in self.agd['phrases'] if d['id'] not in exclude_ids]}
        return agd

    def clear_clusters(self):
        self.clusters = {}
        self.c_id = 0

    def cluster_sentences(self, gesture_data=None, min_cluster_sim=0.5, max_cluster_size=90, max_number_clusters=1000, exclude_gesture_ids=[], include_ids=[]):
        max_number_clusters = max_number_clusters if max_number_clusters else 10000
        if not self.has_assigned_feature_vecs:
            self._assign_feature_vectors()
        self.max_number_clusters = max_number_clusters
        gd = gesture_data if gesture_data else self.agd

        if len(include_ids):
            self.agd['phrases'] = [g for g in self.agd['phrases'] if g['id'] in include_ids]

        # filter here
        if(exclude_gesture_ids):
            gd = self.filter_agd(exclude_gesture_ids)

        i = 0
        l = len(gd)
        print("Clustering sentences, excluding %s gestures" % str(len(exclude_gesture_ids)))

        #double filter?
        phrases = [g for g in gd['phrases'] if g['id'] not in exclude_gesture_ids]

        for g in tqdm(phrases):
            # if 'sentence_embedding' not in g.keys():
            #     g['sentence_embedding'] = self.get_sentence_embedding(g['phase']['transcript'])

            # print "GID: %s" % g['id']
            s = time.time()
            i = i + 1
            # print("finding cluster for gesture %s (%s/%s)" % (g['id'], i, l))
            self._log("finding cluster for gesture %s (%s/%s)" % (g['id'], i, l))
            (nearest_cluster_id, cluster_sim) = self._get_most_similar_cluster_wn(g)
            # we're further away than we're allowed to be, OR this is the first cluster.
            if len(self.clusters) > max_number_clusters:
                # print "%s over max number clusters %s" % (len(self.clusters), max_number_clusters)
                self._add_gesture_to_cluster(g, nearest_cluster_id)
            elif (min_cluster_sim and cluster_sim < min_cluster_sim) or (not len(self.clusters)):
                self._log("creating new cluster for gesture %s -- %s" % (g['id'], i))
                self._create_new_cluster(g)
                g['sentence_cluster_id'] = self.c_id
            else:
                # print ("nearest cluster distance was %s" % cluster_sim)
                st = time.time()
                self._log("fitting in cluster %s" % nearest_cluster_id)
                # print("max cluster sim was %s" % cluster_sim)
                self._add_gesture_to_cluster(g, nearest_cluster_id)
                ed = time.time()
                # print "time to fit the cluster: %s" % str(ed - st)
            if not i % 200:
                self._recluster_singletons()
            e = time.time()
            # print "time to cluster sentence: %s" % str(e-s)
        # now recluster based on where the new centroids are
        print("created %s clusters" % len(self.clusters))
        #self._recluster_by_centroids(phrases)
        # TODO do need some sort of reclustering I think...
        # self._recluster_singletons()
        self._add_sentence_cluster_ids()
        self._add_nearest_cluster()

    def _recluster_by_centroids(self, phrases):
        print("reclustering by centroids")
        for k in self.clusters:
            c = self.clusters[k]
            c['gestures'] = []
            c['sentences'] = []
        for g in tqdm(phrases):
            (nearest_cluster_id, cluster_sim) = self._get_most_similar_cluster(g)
            self._add_gesture_to_cluster(g, nearest_cluster_id)

    def _add_sentence_cluster_ids(self):
        for k in self.clusters:
            c = self.clusters[k]
            for g in c['gestures']:
                g['sentence_cluster_id'] = k
        return

    def _add_nearest_cluster(self):
        keys = list(self.clusters.keys())
        for elem in self.clusters:
            k = keys.pop()
            c = self.clusters[k]
            nearest_cluster_id = ''
            max_sim = 0
            for g in c['gestures']:
                # go through the rest of the keys
                for el in keys:
                    sim = self._get_avg_dist_between_clusts(k, el)
                    if sim > max_sim:
                        max_sim = sim
                        nearest_cluster_id = el
            self.clusters[k]['nearest_cluster_id'] = nearest_cluster_id

    def _get_avg_dist_between_clusts(self, c1_id, c2_id):
        sims = []
        for g in self.clusters[c1_id]['gestures']:
            for g2 in self.clusters[c2_id]['gestures']:
                sims.append(self.get_wn_symmetric_similarity(g['phase']['transcript'], g2['phase']['transcript']))
        avgs = np.array(sims)
        return np.average(avgs)

    def count_sentence_clusters_of_gesture(self, g_id):
        count = 0
        for k in list(self.clusters.keys()):
            c = self.clusters[k]
            for g in c['gestures']:
                if g['id'] == g_id:
                    count += 1
        return count

    def _add_gesture_to_cluster(self, g, cluster_id):
        self.clusters[cluster_id]['gestures'].append(g)
        self.clusters[cluster_id]['sentences'].append(g['phase']['transcript'])
        # if (len(self.clusters[cluster_id]['sentences']) > MAX_CLUSTER_SIZE) and not (len(self.clusters) > max_number_clusters):
        #     self._break_cluster(cluster_id)
        # else:
            # this is like updating the centroid.
        self.clusters[cluster_id]['cluster_embedding'] = self.embed_fn(self.clusters[cluster_id]['sentences'])
        g['sentence_cluster_id'] = cluster_id

    def _break_cluster(self, cluster_id):
        print("breaking up cluster %s" % cluster_id)
        # first, get furthest sentence
        c = self.clusters[cluster_id]
        gs = c['gestures']
        furthest_sentence_g = gs[0]
        min_sim = 10000
        for g in gs:
            sim = self.get_embedded_sentence_similarity_to_cluster(g['sentence_embedding'], cluster_id)
            if sim < min_sim:
                furthest_sentence_g = g
                min_sim = sim
        # then get the 20 sentences closest to THAT sentence.
        gs.sort(key=lambda x: self.get_gesture_sentence_similarity(x, g), reverse=True)
        c1_gs = gs[:(len(gs)/2)]
        c2_gs = gs[(len(gs)/2):]
        self.create_new_cluster_by_gestures(c1_gs)
        self.create_new_cluster_by_gestures(c2_gs)
        del self.clusters[cluster_id]

    def _assign_feature_vectors(self, gesture_data=None):
        gd = gesture_data if gesture_data else self.agd
        print("Getting initial feature vectors.")
        phrases = gd['phrases']
        for g in tqdm(phrases):
            g['sentence_embedding'] = self.get_sentence_embedding(g['phase']['transcript'])
        # I don't think we need to do this because using google's universal thing
        self.has_assigned_feature_vecs = True
        return

    # I think this is eating sentences?
    def _recluster_singletons(self):
        single_cluster_ids = [self.clusters[c]['cluster_id'] for c in list(self.clusters.keys()) if len(self.clusters[c]['sentences']) == 1]
        for single_id in single_cluster_ids:
            (most_sim_cluster_id, sim) = self._get_most_similar_cluster(self.clusters[single_id]['gestures'][0])
            self._add_gesture_to_cluster(self.clusters[single_id]['gestures'][0], most_sim_cluster_id)
            del self.clusters[single_id]

    def create_new_cluster_by_gestures(self, gests):
        new_cluster_id = self.c_id
        sents = [g['phase']['transcript'] for g in gests]
        self.c_id = self.c_id + 1
        for g in gests:
            g['sentence_cluster_id'] = new_cluster_id
        c = {'cluster_id': new_cluster_id,
             'cluster_embedding': self.embed_fn(sents),
             'seed_id': gests[0]['id'],
             'gestures': gests,
             'sentences': sents}
        # e = time.time()
        # print "time to create new cluster: %s" % str(e-s)
        self.clusters[new_cluster_id] = c

    def _create_new_cluster(self, seed_gest):
        s = time.time()
        self._log("creating new cluster for gesture %s" % seed_gest['id'])
        # print("creating new cluster for gesture %s" % seed_gest['id'])
        new_cluster_id = self.c_id
        self.c_id = self.c_id + 1
        c = {'cluster_id': new_cluster_id,
             'cluster_embedding': self.embed_fn([seed_gest['phase']['transcript']]),
             'seed_id': seed_gest['id'],
             'gestures': [seed_gest],
             'sentences': [seed_gest['phase']['transcript']]}
        e = time.time()
        # print "time to create new cluster: %s" % str(e-s)
        self.clusters[new_cluster_id] = c

    def report_clusters(self, verbose=False):
        print(("Number of clusters: %s" % len(self.clusters)))
        num_clusters = len(self.clusters)
        cluster_lengths = [len(self.clusters[c]['sentences']) for c in list(self.clusters.keys())]
        print(("Cluster lengths and ids: %s" % list(zip(list(self.clusters.keys()), cluster_lengths))))
        print(("Avg cluster size: %s" % np.average(cluster_lengths)))
        print(("Median cluster size: %s" % np.median(cluster_lengths)))
        print(("Largest cluster size: %s" % max(cluster_lengths)))
        print(("Sanity check: total clustered gestures: %s / %s" % (sum(cluster_lengths), len(self.agd['phrases']))))
        print(("avg silhouette score:" % self.get_silhouette_scores()))
        # TODO: average and median centroid distances from each other.
        # TODO: also get minimum and maximum centroid distances.
        return self.clusters

    def get_sentences_by_cluster(self, cluster_id):
        return(self.clusters[cluster_id]['sentences'])

    def print_sentences_by_cluster(self, cluster_id):
        sents = self.get_sentences_by_cluster(cluster_id)
        empties = 0
        c = 0
        colors = ['red', 'blue']
        for i, s in enumerate(sents):
            if s:
                print(colored("%s. %s" % (i, s), colors[c]))
                if c:
                    c = 0
                else:
                    c = 1
            else:
                empties += 1
        print("Along with %s empty strings." % empties)
        print()

    ## instead of this need to use centroid.
    def _get_most_similar_cluster(self, g):
        s = time.time()
        max_sim = 0
        nearest_cluster_id = ''
        v = []
        for c in self.clusters:
            v.append(len(self.clusters[c]['sentences']))
        # print "cluster lengths: %s" % v
        for k in self.clusters:
            c = self.clusters[k]
            avg_sims = []
            for s in c['gestures']:
                avg_sims.append(np.inner(g['sentence_embedding'], s['sentence_embedding']).max())
            sim = np.average(np.array(avg_sims))
            if sim > max_sim:
                nearest_cluster_id = k
                max_sim = sim
        e = time.time()
        # print "time to get similar cluster: %s" % str(e-s)
        return (nearest_cluster_id, max_sim)

    def _get_most_similar_cluster_wn(self, g):
        max_sim = -1
        nearest_cluster_id = ''
        for k in self.clusters:
            sim = self._get_avg_similarity_to_cluster(g['phase']['transcript'], k)
            if sim > max_sim:
                nearest_cluster_id = k
                max_sim = sim
        return (nearest_cluster_id, max_sim)

    def _get_avg_similarity_to_cluster(self, sentence, cluster_id):
        c = self.clusters[cluster_id]
        sims = []
        for s in c['sentences']:
            sims.append(self.get_wn_symmetric_similarity(sentence, s))
        return np.average(np.array(sims))

    def count_videos_with_phrase(self, phrase):
        vids = []
        p = self.agd['phrases']
        total = 0
        for g in p:
            if phrase in g['phase']['transcript'].lower():
                vids.append(g['phase']['video_fn'])
                total += 1
        counts = []
        for vid in vids:
            counts.append(vids.count(vid))
        print("total occurances: %s" % total)
        print(len(list(set(vids))))
        return list(set(zip(vids, counts))).sorted(key=lambda x: x[1])

    def find_cluster_ids_for_phrase(self, phrase):
        c_ids = []
        total = 0
        for c in list(self.clusters.keys()):
            count = 0
            for s in self.clusters[c]['sentences']:
                if phrase in s.lower():
                    count += 1
            if count:
                c_ids.append((self.clusters[c]['cluster_id'], count))
                total += count

        print("total occurances: %s" % total)
        return c_ids.sorted(key=lambda x: x[1])

    def _log(self, s):
        self.logs.append(s)

    def _write_logs(self):
        with open(self.logfile, 'w') as f:
            for l in self.logs:
                f.write("%s\n" % l)
        f.close()

    # takes cluster id, returns cluster id of nearest neighbor
    def get_nearest_neighbor_cluster(self, cluster_id):
        max_sim = 0
        nearest_neighbor = list(self.clusters.keys())[0]
        for k in self.clusters:
            if k == cluster_id:
                continue
            sim = np.inner(self.clusters[cluster_id]['cluster_embedding'], self.clusters[k]['cluster_embedding']).max()
            if sim > max_sim:
                max_sim = sim
                nearest_neighbor = k
        return nearest_neighbor

    def get_silhouette_score_for_cluster(self, cluster_id):
        c = self.clusters[cluster_id]
        nearest_neighbor_id = self.get_nearest_neighbor_cluster(cluster_id)
        neighbor_clust = self.clusters[nearest_neighbor_id]
        within_sims = []
        neighbor_sims = []
        for i in range(len(c['gestures'])):
            g = c['gestures'][i]['sentence_embedding']
            for j in range(len(c['gestures'])):
                if i == j:
                    continue
                g2 = c['gestures'][j]['sentence_embedding']
                within_sims.append(np.inner(g, g2).max())
        for i in range(len(c['gestures'])):
            g = c['gestures'][i]['sentence_embedding']
            for j in range(len(neighbor_clust['gestures'])):
                g2 = neighbor_clust['gestures'][j]['sentence_embedding']
                neighbor_sims.append(np.inner(g, g2).max())
        a = np.average(np.array(within_sims))
        b = np.average(np.array(neighbor_sims))
        score = (b - a) / max(b, a)
        return score

    def get_silhouette_scores(self):
        scores = []
        for c in self.clusters:
            scores.append(self.get_silhouette_score_for_cluster(c))
        return np.average(np.array(scores))

    def get_wn_symmetric_similarity(self, s1, s2):
        sim = (get_wordnet_similarity(s1, s2) + get_wordnet_similarity(s2, s1)) / 2
        # TODO make this better
        # if not sim:
        #     print "No similarity found for sentnces \n%s \n %s" % (s1, s2)
        return sim

# TODO come up with way to cache similarity between sentences so it only has to be computed
# once per gesture... does this work though? is there any way around computing similarity between
# EVERY sentence? POTENTIALLY MULTIPLE TIMES??

def get_wordnet_similarity(s1, s2):
    """ compute the sentence similarity using Wordnet """
    # Tokenize and tag
    s1 = pos_tag(word_tokenize(s1))
    s2 = pos_tag(word_tokenize(s2))
    # Get the synsets for the tagged words
    synsets1 = [tagged_to_synset(*tagged_word) for tagged_word in s1]
    synsets2 = [tagged_to_synset(*tagged_word) for tagged_word in s2]
    # Filter out the Nones
    synsets1 = [ss for ss in synsets1 if ss]
    synsets2 = [ss for ss in synsets2 if ss]
    score, count = 0.0, 0
    # For each word in the first sentence
    for synset in synsets1:
        # Get the similarity value of the most similar word in the other sentence
        scores = [synset.path_similarity(ss) for ss in synsets2]
        if len(scores):
            best_score = max(scores)
            # Check that the similarity could have been computed
            if best_score is not None:
                score += best_score
                count += 1
    # Average the values
    if not count:
        # TODO find better metric for this
        return 0
    score /= count
    return score

def penn_to_wn(tag):
    """ Convert between a Penn Treebank tag to a simplified Wordnet tag """
    if tag.startswith('N'):
        return 'n'
    if tag.startswith('V'):
        return 'v'
    if tag.startswith('J'):
        return 'a'
    if tag.startswith('R'):
        return 'r'
    return None

def tagged_to_synset(word, tag):
    wn_tag = penn_to_wn(tag)
    if wn_tag is None:
        return None
    try:
        return wn.synsets(word, wn_tag)[0]
    except:
        return None

