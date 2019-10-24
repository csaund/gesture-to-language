print "loading tf"
import tensorflow as tf
print "loading tfh"
import tensorflow_hub as hub
import random
import numpy as np
from tqdm import tqdm


tf.compat.v1.disable_eager_execution()


# embed = hub.Module("/tmp/sentence-encoder")

# using google's transformer encoder.

## think I'm using this? https://towardsdatascience.com/use-cases-of-googles-universal-sentence-encoder-in-production-dd5aaab4fc15

# Compute a representation for each message, showing various lengths supported.
# elephant = "Elephant"
# dog = "Dog"
# wolf = "Wolf"
# vacuum = "Vacuum"
#
# word = "Elephant"
#
# sentence = "I am a sentence for which I would like to get its embedding."
# sentence1 = "Jazz is a great genre of music."
# sentence2 = "I am a big fan of rock music."
# sentence3 = "It is nice listening to the sounds of nature."
#
#
#
#
# def embed_useT(module):
#     with tf.compat.v1.Graph().as_default():
#         sentences = tf.compat.v1.placeholder(tf.string)
#         embed = hub.Module(module)
#         embeddings = embed(sentences)
#         session = tf.compat.v1.train.MonitoredSession()
#     return lambda x: session.run(embeddings, {sentences: x})
#
# embed_fn = embed_useT("/tmp/sentence-encoder")
#
# paragraph = (
#     "Universal Sentence Encoder embeddings also support short paragraphs. "
#     "There is no hard limit on how long the paragraph is. Roughly, the longer "
#     "the more 'diluted' the embedding will be.")
# # messages = [word, sentence, paragraph]
#
#
# messages = [sentence, sentence1, sentence2, sentence3]
#
# # Reduce logging output.
# tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
# # with tf.compat.v1.Session() as session:
# #     session.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])
# message_embeddings = embed_fn(messages)
# for i, message_embedding in enumerate(np.array(message_embeddings).tolist()):
#         print("Message: {}".format(messages[i]))
#         print("Embedding size: {}".format(len(message_embedding)))
#         message_embedding_snippet = ", ".join((str(x) for x in        message_embedding[:3]))
#         print("Embedding[{},...]\n".
#                    format(message_embedding_snippet))
#
# # print message_embeddings[0]
#
#
# def get_dist(v1, v2):
#     return np.linalg.norm(np.array(v1) - np.array(v2))
#
# print "%s / %s similarity: %s" % (sentence1, sentence2, np.inner(message_embedding[1], message_embedding[2]).max())
# print "%s / %s similarity: %s" % (sentence1, sentence, np.inner(message_embedding[1], message_embedding[0]).max())
# print "%s / %s similarity: %s" % (sentence2, sentence, np.inner(message_embedding[2], message_embedding[0]).max())
# print "%s / %s similarity: %s" % (sentence2, sentence3, np.inner(message_embedding[2], message_embedding[3]).max())
# print "%s / %s similarity: %s" % (sentence, sentence3, np.inner(message_embedding[0], message_embedding[3]).max())
#
#
#
# animals = ["tigers", "lions", "bears"]
# nature = ["trees", "ocean", "rivers"]
# emotions = ['happy', 'sad', 'irritated']
#
# a_mat = embed_fn(animals)
# n_mat = embed_fn(nature)
# e_mat = embed_fn(emotions)
#
# a_test = "birds"
# n_test = "cliff"
# e_test = "joy"
#
# a_test_mat = embed_fn([a_test])
# n_test_mat = embed_fn([n_test])
# e_test_mat = embed_fn([e_test])
#
#
#
# print "Animal test"
# print "Animal similarity: %s " % (np.inner(a_test_mat, a_mat)).max()
# print "Nature similarity: %s " % (np.inner(a_test_mat, n_mat)).max()
# print "Emotion similarity: %s " % (np.inner(a_test_mat, e_mat)).max()
#
# print "Nature test"
# print "Animal similarity: %s " % (np.inner(n_test_mat, a_mat)).max()
# print "Nature similarity: %s " % (np.inner(n_test_mat, n_mat)).max()
# print "Emotion similarity: %s " % (np.inner(n_test_mat, e_mat)).max()
#
# print "Emotion test"
# print "Animal similarity: %s " % (np.inner(e_test_mat, a_mat)).max()
# print "Nature similarity: %s " % (np.inner(e_test_mat, n_mat)).max()
# print "Emotion similarity: %s " % (np.inner(e_test_mat, e_mat)).max()
#
# ## Cluster 6
# c6 = ["farming Community said this",
#     "in the 60s who wants a version of Sergeant Pepper that",
#     "kinds of jobs a songwriter with somebody who",
#     "song after song after song Sometimes in a formula kind of",
#     "rock music where the the the model starts to be for about",
#     "gained a lot of microchip technology that would",
#     "so what happens in popular music is",
#     "shows like one famous one that was hosted by Bing Crosby and of course music lots of music lots",
#     "that that actually moves away from the old Tin Pan Alley idea of a song is being something that you know",
#     "1955 as I was talking before they work with The Coasters some of that some of those playlist that we",
#     "talked about before down in Mexico for 1956 Yakety Yak from 1958 Charlie Brown from",
#     "to know him is to love him interesting ly it seems Phil Spector who wrote that song got the title of the song from an inscription on his father's Tombstone which basically Circus Reno",
#     "installer I started to get into the studio and developed his own approach to",
#     "it was was a big hit for him in 1964 he was actually quite friendly with the Beatles was sitting next to Paul McCartney on the plane when he arrived in JFK with the Beatles came here for the first time getting a lot of career advice I also gave the Rolling Stones live advice came back at the end of The Beatles career 69 to produce that let it",
#     "albums by John Lennon and George Harrison so Phil",
#     "Johnny Mathis that is kind of crooner type black singers who were singing songs that work more sort",
#     "conflicted about it and it was kind of a controversial thing for him to do for a lot of the reasons last week we talked about Little Richard gospel being a being gods music and write rock-and-roller pop being a devil's music Across the way from the church and into saloons and",
#     "bars and this kind of thing but he did make the transition away from gospel in a 1957 had a number one hit with you send me continue to have hits into the",
#     "they want to create something more of lasting value and you can start to see the beginnings of this ambition in there goes my baby will follow the story forward over the",
#     "to be a solo act and was replaced by a felon in brudy Louis with The Drifters so they were really to act Leiber and Stoller work we're working with a working with Benny King as a solo act in The Drifters as a group back with with Benny King I had hits with Spanish Harlem which was a number 10 hit in 1961 and stand by me a number for hit in 1961 both of us having a",
#     "is a version of rockabilly that's a little bit more influenced by the Teen Idol kind of Imaging it's a squeaky or clean kind of rockabilly image there is none of the kind of roughneck Troublemaker part of Rock left at this point is it's really very very sort of squeaky clean in and almost as I say I kind of another version of",
#     "some of those Elvis Presley session when Elvis move from",
#     "Sun to RCA and so that they were assigned to a",
#     "when they shipped it to another label Cathy's clown a number one hit in 1960 and When Will I Be Loved from 1960 and",
#     "I mean let's face it he had a",
#     "roll the end of the 50 and take you to this area that you're the sort of more controlled by the",
#     "neck is the songs they heard when they were a kid not to put the technologies that"]
#
# c2 = ["sings the music of Cole Porter it was really not expected to the song lyrics should",
#     "there was a song",
#     "when it comes to talking about Styles we talked about folk music we talked about country music when",
#     "bit of a debate about that cuz some people thought we got the song is placed in the movie won't that",
#     "a recording studio everybody gets a microphone at the sound of this guy's instrument",
#     "classical fans wouldn't like it in the R&B fans would like it and so he is famous remark was as far as he was concerned There Goes My Baby sounded",
#     "that's that's a little bit where their distinctive vocal style Cubs from they were signed to acuff-rose Publishers in Nashua know",
#     "were they were out"]
#
#
# c6_mat = embed_fn(c6)
# c2_mat = embed_fn(c2)
#
# c6_test = embed_fn([random.choice(c6)])
# c2_test = embed_fn([random.choice(c2)])
#
#
#
# print "C6 test"
# print "C6 similarity: %s " % (np.inner(c6_test, c6_mat)).max()
# print "C2 similarity: %s " % (np.inner(c6_test, c2_mat)).max()
# print "C2 test"
# print "C6 similarity: %s " % (np.inner(c2_test, c6_mat)).max()
# print "C2 similarity: %s " % (np.inner(c2_test, c2_mat)).max()
#
# #
# #
# #
# # print "Done"



class SentenceClusterer():
    def __init__(self):
        print "oh boy we're in for it now."
        self.embed_fn = self.initialize_encoder("/tmp/sentence-encoder")

    def initialize_encoder(self, module):
        with tf.compat.v1.Graph().as_default():
            sentences = tf.compat.v1.placeholder(tf.string)
            embed = hub.Module(module)
            embeddings = embed(sentences)
            session = tf.compat.v1.train.MonitoredSession()
        return lambda x: session.run(embeddings, {sentences: x})


    def encode_sentence(self, item):
        return self.embed_fn(item)


    def cluster_gestures(self, gesture_data=None, max_cluster_distance=0.03):
        if not self.has_assigned_feature_vecs:
            self._assign_feature_vectors()
        gd = gesture_data if gesture_data else self.agd
        i = 0
        l = len(gd)
        print "Clustering sentences"
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
            else:
                self._log("fitting in cluster %s" % nearest_cluster_id)
                self._log("nearest cluster distance was %s" % nearest_cluster_dist)
                self.clusters[nearest_cluster_id]['gestures'].append(g)
                self._update_cluster_centroid(nearest_cluster_id)
            end = time.time()
            self._log(str(end-start))

        # now recluster based on where the new centroids are
        self._recluster_by_centroids()

        self._write_logs()

    # takes two strings
    def get_sentence_embedding(self, s):
        return embed_fn([s])

    # takes two strings
    def get_sentence_similarity(self, s1, s2):
        s1_mat = embed_fn([s1])
        s2_mat = embed_fn([s2])
        return np.inner(s1_mat, s2_mat).max()


    def get_sentence_similarity_to_paragraph(self, s, p):
        s_mat = embed_fn([s])
        p_mat = embed_fn(p)
        return np.inner(s_mat, p_mat).max()

    def get_embedded_sentence_similarity_to_paragraph(self, s_mat, p):
        p_mat = embed_fn(p)
        return np.inner(s_mat, p_mat).max()

    def get_embedded_sentence_similarity_to_cluster(self, s_mat, cluster_id):
        p_mat = self.clusters['cluster_id']['centroid']
        return np.inner(s_mat, p_mat).max()


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
            (nearest_cluster_id, nearest_cluster_dist) = self._get_most_similar_cluster(g)
            # we're further away than we're allowed to be, OR this is the first cluster.
            if (max_cluster_distance and nearest_cluster_dist > max_cluster_distance) or (not len(self.clusters)):
                self._log("nearest cluster distance was %s" % nearest_cluster_dist)
                self._log("creating new cluster for gesture %s -- %s" % (g['id'], i))
                self._create_new_cluster(g)
                g['cluster_id'] = self.c_id
            else:
                self._log("fitting in cluster %s" % nearest_cluster_id)
                self._log("nearest cluster distance was %s" % nearest_cluster_dist)
                self.clusters[nearest_cluster_id]['gestures'].append(g)
                self._update_cluster_centroid(nearest_cluster_id)
                g['cluster_id'] = neartest_cluster_id
            end = time.time()
            self._log(str(end-start))

        # now recluster based on where the new centroids are
        self._recluster_by_centroids()

        self._write_logs()



    def _assign_feature_vectors(self, gesture_data=None):
        gd = gesture_data if gesture_data else self.agd
        print "Getting initial feature vectors."
        for g in tqdm(gd):
            g['sentence_feature_vec'] = self._get_gesture_features(g)
        self._normalize_feature_values()
        self.has_assigned_feature_vecs = True
        return

    def _normalize_feature_values(self, gesture_data=None):
        print "Normalizing feature vectors."
        gd = gesture_data if gesture_data else self.agd
        feat_vecs = np.array([g['sentence_feature_vec'] for g in self.agd])
        feats_norm = np.array(map(lambda v: v / np.linalg.norm(v), feat_vecs.T))
        feat_vecs_normalized = feats_norm.T
        print "Reassigning normalized vectors"
        for i in tqdm(range(len(gd))):
            gd[i]['sentence_feature_vec'] = list(feat_vecs_normalized[i])
        return


    def _create_new_cluster(self, seed_gest):
        self._log("creating new cluster for gesture %s" % seed_gest['id'])
        new_cluster_id = self.c_id
        self.c_id = self.c_id + 1
        c = {'cluster_id': new_cluster_id,
             'centroid': seed_gest['sentence_feature_vec'],
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
                d = self._calculate_distance_between_vectors(g['sentence_feature_vec'], self.clusters[c]['centroid'])
                if d < min_dist:
                    min_dist = d
                    min_clust = self.clusters[c]
            min_clust['gestures'].append(g)
        return

    ## instead of this need to use centroid.
    def _get_most_similar_cluster(self, g):
        max_sim = 0
        nearest_cluster_id = ''
        for k in self.clusters:
            sim = self.get_embedded_sentence_similarity_to_cluster(g['sentence_feature_vec'], k)
            if sim > max_sim:
                nearest_cluster_id = k
                sim = max_sim
        return (nearest_cluster_id, shortest_dist)
