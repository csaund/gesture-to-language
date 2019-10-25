print "loading tf"
import tensorflow as tf
print "loading tfh"
import tensorflow_hub as hub
import random
import numpy as np
from tqdm import tqdm
from common_helpers import *
from termcolor import colored

import time

print "loading nltk"
## TODO: use this?
from nltk.corpus import wordnet as wn
from nltk import sentiment as sent

tf.compat.v1.disable_eager_execution()

MAX_CLUSTER_SIZE = 50
CLUSTER_SIMILARITY_TOLERANCE = 0.6

# from NewSentenceClusterer import *
# SC = SentenceClusterer("/Users/carolynsaund/github/gest-data/data", "rock")
# SC.cluster_sentences(gd)

class SentenceClusterer():
    def __init__(self, base_path, speaker, seeds=[]):
        self.speaker = speaker
        self.base_path = base_path
        print "oh boy we're in for it now."
        self.embed_fn = self.initialize_encoder("/tmp/sentence-encoder")
        self.has_assigned_feature_vecs = False
        self.logfile = "/Users/carolynsaund/github/gesture-to-language/sentence_cluster_logs.txt"
        self.cluster_file = "/Users/carolynsaund/github/gesture-to-language/sentence_cluster_tmp.json"
        self.full_transcript_bucket = "full_timings_with_transcript_bucket"
        self.clusters = {}
        self.logs = []
        if(len(seeds)):
            for seed_g in seeds:
                g = self._get_gesture_by_id(seed_g, all_gesture_data)
                cluster_id = self.c_id
                self.c_id = self.c_id + 1
                c = {
                        'cluster_id': cluster_id,
                        'seed_id': g['id'],
                        'gestures': [g['id']],
                        'sentences': [g['phase']['transcript']]}
                self.clusters[cluster_id] = c
        self.agd = None
        self.c_id = 0
        self.get_transcript()

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


    def cluster_sentences(self, gesture_data=None, min_cluster_sim=0.5):
        # if not self.has_assigned_feature_vecs:
        #     self._assign_feature_vectors()
        gd = gesture_data if gesture_data else self.agd
        if ('sentence_embedding' not in gd['phrases'][0].keys()):
            self._assign_feature_vectors()
        i = 0
        l = len(gd)
        phrases = gd['phrases']
        print "Clustering sentences"
        for g in tqdm(phrases):
            # print "GID: %s" % g['id']
            s = time.time()
            i = i + 1
            # print("finding cluster for gesture %s (%s/%s)" % (g['id'], i, l))
            self._log("finding cluster for gesture %s (%s/%s)" % (g['id'], i, l))
            (nearest_cluster_id, cluster_sim) = self._get_most_similar_cluster(g)
            # we're further away than we're allowed to be, OR this is the first cluster.
            if (min_cluster_sim and cluster_sim < min_cluster_sim) or (not len(self.clusters)):
                # print ("nearest cluster distance was %s" % cluster_sim)
                self._log("creating new cluster for gesture %s -- %s" % (g['id'], i))
                self._create_new_cluster(g)
                g['sentence_cluster_id'] = self.c_id
            else:
                st = time.time()
                self._log("fitting in cluster %s" % nearest_cluster_id)
                # print("max cluster sim was %s" % cluster_sim)
                self._add_gesture_to_cluster(g, nearest_cluster_id)
                ed = time.time()
                # print "time to fit the cluster: %s" % str(ed - st)
            e = time.time()
            # print "time to cluster sentence: %s" % str(e-s)
        # now recluster based on where the new centroids are
        # self._recluster_by_centroids()
        # TODO do need some sort of reclustering I think...
        self._recluster_singletons()
        self._write_logs()

    def _add_gesture_to_cluster(self, g, cluster_id):
        self.clusters[cluster_id]['gestures'].append(g)
        self.clusters[cluster_id]['sentences'].append(g['phase']['transcript'])
        if len(self.clusters[cluster_id]['sentences']) > MAX_CLUSTER_SIZE:
            self._break_cluster(cluster_id)
        else:
            # this is like updating the centroid.
            self.clusters[cluster_id]['cluster_embedding'] = self.embed_fn(self.clusters[cluster_id]['sentences'])
            g['sentence_cluster_id'] = cluster_id


    def _break_cluster(self, cluster_id):
        print "breaking up cluster %s" % cluster_id
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
        sorted(gs, key=lambda x: self.get_gesture_sentence_similarity(x, g), reverse=True)
        c1_gs = gs[:(len(gs)/2)]
        c2_gs = gs[(len(gs)/2):]
        self._create_new_cluster_by_gestures(c1_gs)
        self._create_new_cluster_by_gestures(c2_gs)
        del self.clusters[cluster_id]

    def _assign_feature_vectors(self, gesture_data=None):
        gd = gesture_data if gesture_data else self.agd
        print "Getting initial feature vectors."
        phrases = gd['phrases']
        for g in tqdm(phrases):
            g['sentence_embedding'] = self.get_sentence_embedding(g['phase']['transcript'])
        # I don't think we need to do this because using google's universal thing
        self.has_assigned_feature_vecs = True
        return


    def _recluster_singletons(self):
        single_cluster_ids = [self.clusters[c]['cluster_id'] for c in self.clusters.keys() if len(self.clusters[c]['sentences']) == 1]
        for single_id in single_cluster_ids:
            (most_sim_cluster_id, sim) = self._get_most_similar_cluster(self.clusters[single_id]['gestures'][0])
            self._add_gesture_to_cluster(self.clusters[single_id]['gestures'][0], most_sim_cluster_id)
            del self.clusters[single_id]


    def _create_new_cluster_by_gestures(self, gests):
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


    def find_cluster_ids_for_phrase(self, phrase):
        c_ids = []
        for c in self.clusters.keys():
            count = 0
            for s in self.clusters[c]['sentences']:
                if phrase in s.lower():
                    count += 1
            if count:
                c_ids.append((self.clusters[c]['cluster_id'], count))
        return c_ids

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
        print("Number of clusters: %s" % len(self.clusters))
        num_clusters = len(self.clusters)
        cluster_lengths = [len(self.clusters[c]['sentences']) for c in self.clusters.keys()]
        print("Cluster lengths and ids: %s" % zip(self.clusters.keys(), cluster_lengths))
        print("Avg cluster size: %s" % np.average(cluster_lengths))
        print("Median cluster size: %s" % np.median(cluster_lengths))
        print("Largest cluster size: %s" % max(cluster_lengths))
        print("Sanity check: total clustered gestures: %s / %s" % (sum(cluster_lengths), len(self.agd['phrases'])))
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
                print colored("%s. %s" % (i, s), colors[c])
                if c:
                    c = 0
                else:
                    c = 1
            else:
                empties += 1
        print "Along with %s empty strings." % empties
        print

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
            sim = np.inner(g['sentence_embedding'], self.clusters[k]['cluster_embedding']).max()
            if sim > max_sim:
                nearest_cluster_id = k
                max_sim = sim
        e = time.time()
        # print "time to get similar cluster: %s" % str(e-s)
        return (nearest_cluster_id, max_sim)



    def _log(self, s):
        self.logs.append(s)

    def _write_logs(self):
        with open(self.logfile, 'w') as f:
            for l in self.logs:
                f.write("%s\n" % l)
        f.close()

    ## TODO make use of these?
    #############################################################
    ####### EVERYTHING BELOW HERE IS NOT IN USE YET #############
    def wnexpand(set):
          res=Set(set)
          #print res
          lst = []
          for w in set:
           for ss in wn.synsets(morph(w)):
             top = Set(ss.lemma_names())
             res = res.union(top)
             for sim in ss.similar_tos():
                 res=res.union(Set(sim.lemma_names()))
          for u in res:
           lst.append(u.encode('ascii','ignore'))
          return lst


    def morph(w0):
          u = wn.morphy(str(w0))
          if (u == None):
           #print w0
           return w0
          else:
           w = u.encode('ascii','ignore')
           print w
           return w

    def get_hypernyms(w0):
        syn = wn.synsets(w0)

        ## dunno when TF this happens
        if type(syn) != list:
            return syn.name()
        ## sometimes it's an empty list??
        elif len(syn) == 0:
            return []

        # most of the time I want hypernyms tho
        hyp_list = list(set([hy.name().split('.')[0] for hy in syn]))
        return hyp_list


    def get_sentence_sentiment_vector(self, sent, sentiment_model):
        sent_vec =[]
        numw = 0
        for w in sent:
            try:
                if numw == 0:
                    sent_vec = sentiment_model[w]
                else:
                    sent_vec = np.add(sent_vec, sentiment_model[w])
                numw+=1
            except:
                pass
        return np.asarray(sent_vec) / numw


    def cluster_by_sentiment(self, transcript=None):
        print "clustering by sentiment"
        trans = transcript if transcript else self.transcript_with_timings
        sentences = self.get_sentences(trans)
        sentences = self._drop_empties(sentences)
        sentiment_model = Word2Vec(sentences, min_count=1)
        X= map(lambda s: self.get_sentence_sentiment_vector(s, sentiment_model), sentences)

        kclusterer = KMeansClusterer(NUM_CLUSTERS, distance=nltk.cluster.util.cosine_distance, repeats=25)
        assigned_clusters = kclusterer.cluster(X, assign_clusters=True)

        kmeans_sentiment = cluster.KMeans(n_clusters=NUM_CLUSTERS)
        kmeans_sentiment.fit(X)

        labels = kmeans_sentiment.labels_
        centroids = kmeans_sentiment.cluster_centers_

        silhouette_score = metrics.silhouette_score(X, labels, metric='euclidean')
        model = TSNE(n_components=2, random_state=0)
        np.set_printoptions(suppress=True)
        Y=model.fit_transform(X)
        plt.scatter(Y[:, 0], Y[:, 1], c=assigned_clusters, s=290,alpha=.5)

        for j in range(len(sentences)):
           plt.annotate(assigned_clusters[j],xy=(Y[j][0], Y[j][1]),xytext=(0,0),textcoords='offset points')
           print ("%s %s" % (assigned_clusters[j],  sentences[j]))

        return kmeans_sentiment

    def _drop_empties(self, v):
        print "Dropping %s empty strings." % v.count('')
        return [i for i in v if i != '']

    def cluster_by_tfidf(self, transcript):
        print "clustering by hypernym"
        hypernyms = self.get_hypernyms(transcript)

        hypes_list = [x['hypernyms'] for x in hypernyms]
        joined_lists = [' '.join(l) for l in hypes_list]

        tfidf_vectorizer = TfidfVectorizer()
        tfidf = tfidf_vectorizer.fit_transform(joined_lists)

        tdidf_kmeans = KMeans(n_clusters=NUM_CLUSTERS).fit(tfidf)

        return tdidf_kmeans


def _add_gesture_to_cluster(sc, g, cluster_id):
    sc.clusters[cluster_id]['gestures'].append(g)
    sc.clusters[cluster_id]['sentences'].append(g['phase']['transcript'])
    if len(sc.clusters[cluster_id]['sentences']) > MAX_CLUSTER_SIZE:
        sc._break_cluster(cluster_id)
    else:
        # this is like updating the centroid.
        sc.clusters[cluster_id]['cluster_embedding'] = sc.embed_fn(sc.clusters[cluster_id]['sentences'])
        g['sentence_cluster_id'] = cluster_id


def _recluster_singletons(sc):
    clusters = sc.clusters
    single_cluster_ids = [sc.clusters[c]['cluster_id'] for c in sc.clusters.keys() if len(sc.clusters[c]['sentences']) == 1]
    for single_id in single_cluster_ids:
        (most_sim_cluster_id, sim) = sc._get_most_similar_cluster(sc.clusters[single_id]['gestures'][0])
        _add_gesture_to_cluster(sc, sc.clusters[single_id]['gestures'][0], most_sim_cluster_id)
        del sc.clusters[single_id]


def find_cluster_ids_for_phrase(sc, phrase):
    c_ids = []
    for c in sc.clusters.keys():
        count = 0
        for s in sc.clusters[c]['sentences']:
            if phrase in s.lower():
                count += 1
        if count:
            c_ids.append((sc.clusters[c]['cluster_id'], count))
    return c_ids
