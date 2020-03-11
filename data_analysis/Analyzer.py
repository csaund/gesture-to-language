from common_helpers import *
from tqdm import tqdm
from scipy import stats
import matplotlib.pyplot as plt
import random

# mathematically find maximum matches aka how much does S1 matter to G1 and G1 matter to S1
# does all the analysis, NO data manipulation in here!!
class Analyzer():
    def __init__(self, gsm):
        self.SentenceClusterer = gsm.SentenceClusterer
        self.GestureClusterer = gsm.GestureClusterer
        self.complete_gesture_data = gsm.complete_gesture_data

    def LOOCV_gesture_mappings(self):
        similarities = []
        for gid in tqdm(self.complete_gesture_data.keys()):
            g = self.complete_gesture_data[gid]
            s = g['sentence_embedding']
            print "sentences: %s" % g['phase']['transcript']
            g_features = g['feature_vec']
            max_sim = 0
            s_max_id = 0
            for gid2 in self.complete_gesture_data.keys():
                if gid2 == gid:
                    continue
                g2 = self.complete_gesture_data[gid2]
                s_prime = g2['sentence_embedding']
                sim = np.inner(s, s_prime).max()
                if sim > max_sim:
                    max_sim = sim
                    s_max_id = gid2
                    print "new max: %s ; %s" % (max_sim, g2['phase']['transcript'])
            print
            # now get similarity between gestures.
            highest_match_gesture_vec = self.complete_gesture_data[s_max_id]['feature_vec']
            gesture_dist = calculate_distance_between_vectors(g_features, highest_match_gesture_vec)
            similarities.append(gesture_dist)
        print "average gesture distance for matching sentences: %s" % np.average(np.array(similarities))
        return np.average(np.array(similarities))

    def get_avg_gesture_distance(self):
        vectors = [self.complete_gesture_data[k]['feature_vec'] for k in self.complete_gesture_data.keys()]
        dists = []
        for i in tqdm(range(len(vectors))):
            # don't want to redo ones we've already done
            for j in range(i, len(vectors)):
                if i == j:
                    continue
                dists.append(calculate_distance_between_vectors(vectors[i], vectors[j]))
        print "average gesture distances: %s" % np.average(np.array(dists))
        return dists

    def get_mapped_to_avg_distance_ratio(self):
        avg_vector_dist = self.get_avg_gesture_distance()
        avg_mapped_dist = self.LOOCV_gesture_mappings()
        print "average distance: %s" % avg_vector_dist
        print "average mapped: %s" % avg_mapped_dist
        print "ratio: %s" % str(float(avg_mapped_dist / avg_vector_dist))
        return float(avg_mapped_dist / avg_vector_dist)

    def get_avg_sentence_dist(self):
        keys = self.complete_gesture_data.keys()
        similarities = []
        for i in tqdm(range(len(keys))):
            s = self.complete_gesture_data[keys[i]]['sentence_embedding']
            for j in (range(i, len(keys))):
                if i == j:
                    continue
                s_j = self.complete_gesture_data[keys[j]]['sentence_embedding']
                similarities.append(np.inner(s, s_j).max())
        print "average sentence similarity: %s" % np.average(np.array(similarities))
        return np.average(np.array(similarities))

    ## TODO this but for wordnet
    def compare_avg_sentence_dist_with_avg_centroid_dist(self, s_cluster_id):
        s_clust = self.SentenceClusterer.clusters[s_cluster_id]
        cluster_embedding = s_clust['cluster_embedding']
        avg_sentence_dist = []
        avg_sentence_to_centroid_dist = []
        for i in range(len(s_clust['gestures'])):
            sentence_embedding = s_clust['gestures'][i]['sentence_embedding']
            avg_sentence_to_centroid_dist.append(np.inner(cluster_embedding, sentence_embedding).max())
            for j in range(i, len(s_clust['gestures'])):
                avg_sentence_dist.append(
                    np.inner(sentence_embedding, s_clust['gestures'][j]['sentence_embedding']).max())
        print "average similarity to cluster embedding: %s" % np.average(np.array(avg_sentence_to_centroid_dist))
        print "average sentence similarity: %s" % np.average(np.array(avg_sentence_dist))

    ## this should basically be a repeat of what I've already done
    def get_closest_gesture_by_sentence(self, gid):
        g = self.complete_gesture_data[gid]
        max_sim = 0
        max_gesture_id = 0
        for comp in self.complete_gesture_data.keys():
            if comp == gid:
                continue
            sim = np.inner(g['sentence_embedding'], self.complete_gesture_data[comp]['sentence_embedding']).max()
            if sim > max_sim:
                max_sim = sim
                max_gesture_id = comp
        return max_gesture_id

    def get_avg_closest_gesture_by_sentences(self):
        avg_dists = []
        for g in tqdm(self.complete_gesture_data.keys()):
            gest = self.complete_gesture_data[g]
            closest = self.get_closest_gesture_by_sentence(g)
            avg_dists.append(calculate_distance_between_vectors(gest['feature_vec'], closest['feature_vec']))
        print "avg matching sentence gesture distance: %s" % np.average(np.array(avg_dists))
        print "min matching sentence gesture distance: %s" % np.min(np.array(avg_dists))
        print "max matching sentence gesture distance: %s" % np.max(np.array(avg_dists))
        print "median matching sentence gesture distance: %s" % np.median(np.array(avg_dists))
        print "sd matching sentence gesture distance: %s" % np.std(np.array(avg_dists))
        return avg_dists

    def get_avg_to_mapped_ratio(self):
        mapped_dists = self.get_avg_closest_gesture_by_sentences()
        avg_dists = self.get_avg_gesture_distance()
        t, p = stats.ttest_ind(mapped_dists, avg_dists)
        print "Mapped vs average distance are different:"
        print "t = %s" % str(t)
        print "p = %s" % str(p)
        ratios = np.array(mapped_dists) / np.average(np.array(avg_dists))
        print "Ratio of mapped/unmapped sig diff from 0"
        t_one_samp, p_one_samp = stats.ttest_1samp(ratios, 0)
        print "t = %s" % str(t_one_samp)
        print "p = %s" % str(p_one_samp)
        print "Ratio of mapped/unmapped sig diff from 1"
        t_one_samp, p_one_samp = stats.ttest_1samp(ratios, 1)
        print "t = %s" % str(t_one_samp)
        print "p = %s" % str(p_one_samp)

    # Plot distance between S and S' on X,
    # g and g' on Y
    def plot_dist_s_sprime_g_gprime(self):
        sentence_similarities = []
        gesture_distances = []
        keys = self.complete_gesture_data.keys()
        for i in range(len(keys)):
            s = self.complete_gesture_data[keys[i]]['sentence_embedding']
            for j in tqdm((range(i, len(keys)))):
                if i == j:
                    continue
                s_j = self.complete_gesture_data[keys[j]]['sentence_embedding']
                sentence_similarities.append(np.inner(s, s_j).max())
                g1fv = self.complete_gesture_data[keys[i]]['feature_vec']
                g2fv = self.complete_gesture_data[keys[j]]['feature_vec']
                gesture_distances.append(calculate_distance_between_vectors(g1fv, g2fv))
        plt.scatter(sentence_similarities, gesture_distances)
        plt.title('Sentence Similarity vs Gesture Distance')
        plt.xlabel('Sentence Similarity')
        plt.ylabel('Gesture Distance')
        plt.show()

    def get_random_gesture_from_sentence_cluster(self, s_cluster_id):
        g = random.choice(self.SentenceClusterer.clusters[s_cluster_id]['gestures'])
        return self.complete_gesture_data[g['id']]

    def plot_semantics_single_gesture(self, gid):
        g = self.complete_gesture_data[gid]
        sentence_similarities = []
        gesture_distances = []
        s = g['sentence_embedding']
        for k in tqdm(self.complete_gesture_data):
            if k == gid:
                continue
            comp = self.complete_gesture_data[k]
            s_j = comp['sentence_embedding']
            sentence_similarities.append(np.inner(s, s_j).max())
            g1fv = g['feature_vec']
            g2fv = comp['feature_vec']
            gesture_distances.append(calculate_distance_between_vectors(g1fv, g2fv))
        plt.scatter(sentence_similarities, gesture_distances)
        plt.xlabel('Sentence Similarity')
        plt.ylabel('Gesture Distance')
        sentence_similarites = np.array(sentence_similarities)
        gesture_distances = np.array(gesture_distances)
        linreg = stats.linregress(sentence_similarites, gesture_distances)
        slope, intercept, r_value, p_value, std_err = linreg
        ti = 'Sentence Similarity vs Gesture Distance for gesture %s \n r2 = %s' % (str(sentence_similarities['id']), r_value ** 2)
        plt.title(ti)
        plt.plot(sentence_similarites, intercept + slope * gesture_distances, 'r')
        plt.text(0, 1, r_value ** 2)
        plt.show()

    def get_semantically_distinguishable_gestures(self, take_n=0):
        # TODO take out beat gesture cluster/large sentence cluster
        take_n = take_n if take_n else len(self.complete_gesture_data)
        totals = []
        keys = self.complete_gesture_data.keys()
        for i in tqdm(range(take_n)):
            k = keys[i]
            g = self.complete_gesture_data[k]
            s = g['sentence_embedding']
            sentence_similarities = []
            gesture_distances = []
            for j in keys:      # TODO also take out beat gesture here
                if k == j:
                    continue
                s_j = self.complete_gesture_data[j]['sentence_embedding']
                sentence_similarities.append(np.inner(s, s_j).max())
                g1fv = self.complete_gesture_data[k]['feature_vec']
                g2fv = self.complete_gesture_data[j]['feature_vec']
                gesture_distances.append(np.linalg.norm(np.array(g1fv) - np.array(g2fv)))
            sentence_similarities = np.array(sentence_similarities)
            gesture_distances = np.array(gesture_distances)
            linreg = stats.linregress(sentence_similarities, gesture_distances)
            slope, intercept, r_value, p_value, std_err = linreg
            totals.append((r_value ** 2, sentence_similarities, gesture_distances, k))
        return sorted(totals, key=lambda x: x[0], reverse=True)

    def get_semantics_single_gesture(self, gid):
        g = self.complete_gesture_data[gid]
        sentence_similarities = []
        gesture_distances = []
        s = g['sentence_embedding']
        for k in tqdm(self.complete_gesture_data):
            if k == gid:
                continue
            comp = self.complete_gesture_data[k]
            s_j = comp['sentence_embedding']
            sentence_similarities.append(np.inner(s, s_j).max())
            g1fv = g['feature_vec']
            g2fv = comp['feature_vec']
            gesture_distances.append(calculate_distance_between_vectors(g1fv, g2fv))
        plt.scatter(sentence_similarities, gesture_distances)
        plt.xlabel('Sentence Similarity')
        plt.ylabel('Gesture Distance')
        sentence_similarities = np.array(sentence_similarities)
        gesture_distances = np.array(gesture_distances)
        return (sentence_similarities, gesture_distances)

    def plot_semantics_for_gesture(self, gid):
        (sentence_similarities, gesture_distances) = self.get_semantics_single_gesture(gid)
        linreg = stats.linregress(sentence_similarities, gesture_distances)
        slope, intercept, r_value, p_value, std_err = linreg
        ti = 'Sentence Similarity vs Gesture Distance for gesture %s \n r2 = %s' % (str(gid), r_value ** 2)
        plt.title(ti)
        plt.scatter(sentence_similarities, gesture_distances)
        plt.xlabel('Sentence Similarity')
        plt.ylabel('Gesture Distance')
        plt.plot(sentence_similarities, intercept + slope * gesture_distances, 'r')
        plt.text(0, 1, r_value ** 2)
        plt.show()

    def plot_dist_s_sprime_g_gprime_wn(self):
        sentence_similarities = []
        gesture_distances = []
        keys = self.complete_gesture_data.keys()
        for i in range(len(keys)):
            s = self.complete_gesture_data[keys[i]]['sentence_embedding']
            for j in tqdm((range(i, len(keys)))):
                if i == j:
                    continue
                s_j = self.complete_gesture_data[keys[j]]['sentence_embedding']
                sentence_similarities.append(np.inner(s, s_j).max())
                g1fv = self.complete_gesture_data[keys[i]]['feature_vec']
                g2fv = self.complete_gesture_data[keys[j]]['feature_vec']
                gesture_distances.append(calculate_distance_between_vectors(g1fv, g2fv))
        plt.scatter(sentence_similarities, gesture_distances)
        plt.title('Sentence Similarity vs Gesture Distance')
        plt.xlabel('Sentence Similarity')
        plt.ylabel('Gesture Distance')
        plt.show()


def t_test(g1, g2):
    return stats.ttest_ind(g1, g2)