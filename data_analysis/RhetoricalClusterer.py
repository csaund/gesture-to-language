from subprocess import call
import re
import os
from common_helpers import download_blob
from tqdm import tqdm
import string

import numpy as np
from sklearn.cluster import AgglomerativeClustering, DBSCAN
import distance
from collections import Counter
import edlib
import sklearn
import matplotlib.pyplot as plt


## Takes the rhetorical parse located in gs://parsed_transcript_bucket
## Determines similarity of two rhetorical structures of a gesture
KEY_TERMS = {
    "Root": "Ro",
    "Nucleus": "Nu",
    "Satellite": "Sa",
    "leaf": "Le",
    "span": "Sp",
    "Elaboration": "El",
    "Attribution": "At",
    "Same-Unit": "Su",
    "Cause": "Ca",
    "Temporal": "Te",
    "Background": "Ba",
    "Enablement": "En",
    "Joint": "Jo",
    "Contrast": "Co",
    "Condition": "Cd",
    "Manner-Means": "Mm",
    "Explanation": "Ex",
    "Comparison": "Cm",
    "text": "Tx"
}

VID_EXTENSION_REPLACEMENTS = {
    'mkv': 'json.rhet_parse',
    'webm': 'json.rhet_parse',
    'mp4': 'json.rhet_parse'
}


PUNCTUATION_REPLACEMENTS = {
    " '": "'",
    " n't": "n't",
    " .": ".",
    " %": "%",
    "$ ": "$",
    ". * ": " * "
}

PARSER_REPLACEMENTS = {
    "n't": " nt",
    "'s": " s",
    "'m": " m",
    "'re": " re",
    "'ve": " ve",
    "'ll": " ll",
    "'": " ",
}

# SOURCES
# https://gist.github.com/codehacken/8b9316e025beeabb082dda4d0654a6fa


def flatten(li): return flatten(li[0]) + (flatten(li[1:]) if len(li) > 1 else []) if type(li) is list else [li]


def multi_index(li, val): return [i for i in range(len(li)) if li[i] == val]


# https://stackoverflow.com/questions/15175142/how-can-i-do-multiple-substitutions-using-regex-in-python
def multi_replace(text, replacement_dict=None):
    if replacement_dict is None:
        replacement_dict = PUNCTUATION_REPLACEMENTS
    try:
        regex = re.compile("(%s)" % "|".join(map(re.escape, replacement_dict.keys())))
        # For each match, look-up corresponding value in dictionary
        return regex.sub(lambda mo: replacement_dict[mo.string[mo.start():mo.end()]], text)
    except:
        print("could not multi replace text with dict")
        print(text)
        print(replacement_dict)


def get_parse_data(fp):
    f = open(fp, "r")
    rhet_lines = f.readlines()
    f.close()
    return rhet_lines


def get_line_encoding(line):
    encoding = "("
    for k in KEY_TERMS.keys():
        if k in line:
            encoding += KEY_TERMS[k]
    encoding += ")"
    return encoding


def get_text_splices(line):
    if "_!" in line:
        text = "".join(line.split("_!")[-2].split("'"))
        return multi_replace(text)
    return ""


def get_sequence_encoding(content=None, rhet_file=None):
    content = content if content else get_parse_data(rhet_file)
    texts = []
    encoding = ""
    for line in content:
        temp = get_line_encoding(line)
        if temp != "()":
            texts.append(get_text_splices(line))
            encoding += temp
    en = ("".join(encoding.split("(")[1:])).split(")")
    return en, texts


def get_matching_words(transcript, texts):
    words = multi_replace(transcript, PARSER_REPLACEMENTS).split(" ")
    words = [w.translate(str.maketrans('', '', string.punctuation)) for w in words]
    word_counter = 0
    text_indexes = []
    working_chunks = []
    for t_index in range(len(texts)):
        t = texts[t_index]
        if t == "":     # skip all blank ones.
            continue
        chunk = t.split(" ")
        cs = []
        for c in chunk:
            cs.append(c.translate(str.maketrans('', '', string.punctuation)))
            working_chunks.append(c.translate(str.maketrans('', '', string.punctuation)))
        working_chunks.append("/")
        # our word isn't in this chunk, start over
        if words[word_counter].translate(str.maketrans('', '', string.punctuation)) not in cs:
            word_counter = 0
            text_indexes = []
            working_chunks = []

        # check that this one isn't the start of the correct block
        if words[word_counter].translate(str.maketrans('', '', string.punctuation)) not in cs:
            continue
        else:       # our word IS in this chunk!
            word_indexes = multi_index(cs, words[word_counter])
            successful_chunk_found = False
            for wi in word_indexes:
                if try_index(words, word_counter, chunk, wi):
                    text_indexes.append(t_index)
                    successful_chunk_found = True
                    word_counter += len(chunk) - wi      # move up to the next place in line
                    if word_counter >= len(words)-1:
                        return sorted(list(set(text_indexes))), " ".join(working_chunks).split("/")[:-1]
                    break
            if not successful_chunk_found:
                word_counter = 0
                text_indexes = []
                working_chunks = []

    return sorted(list(set(text_indexes))), " ".join(working_chunks).split("/")[:-1]


# see if words come in order in a chunk starting from chunk_index
def try_index(words, word_index, chunk, chunk_index):
    if chunk_index >= len(chunk):
        return False
    while chunk_index < len(chunk) and word_index < len(words):
        if words[word_index].translate(str.maketrans('', '', string.punctuation)) == chunk[chunk_index].translate(str.maketrans('', '', string.punctuation)):
            word_index += 1
            chunk_index += 1
        else:
            return False
    return True


def get_rhetorical_encoding_for_gesture(g):
    transcript_to_match = g['phase']['transcript']
    rhetorical_parse_file = multi_replace(g['phase']['video_fn'], VID_EXTENSION_REPLACEMENTS)
    en, texts = get_sequence_encoding(rhet_file=rhetorical_parse_file)
    text_range, text_chunks = get_matching_words(transcript_to_match, texts)
    if not text_range:
        print("No rhetorical encoding found for gesture ", g['id'])
        return None
    return en[text_range[0]:text_range[-1]+1]


def sort_indexes(el):
    return str(el[0]).replace("-", ".")


def get_lev_similarities_by_rhetorical_units(df):
    print("getting edit distances")
    words = []
    order = list(zip(df.id, df.rhetorical_units))  # keep dict in order to sort and
    for k, v in sorted(order, key=sort_indexes):  # assign proper distances to it.
        if not v:
            words.append("")
        elif isinstance(v, dict):
            words.append(v['sequence'])
        else:
            sequences = [el['sequence'] for el in v]
            words.append(" ".join(sequences))
    lev_similarity = []
    for i in tqdm(range(len(words))):
        w = words[i]
        lev_similarity.append([edlib.align(w, w2)['editDistance'] for w2 in words])
    similarities = np.array(lev_similarity)
    return similarities


def get_dbscan_clustering(similarities):
    clustering = DBSCAN(metric='precomputed')
    # similarities is nxn matrix (lev_sim)
    u = clustering.fit_predict(similarities)
    print(sklearn.metrics.silhouette_score(similarities, clustering.labels_))
    counts = Counter(u)
    lens = np.array([counts[c] for c in counts])
    print("max size: ", max(lens))
    print("std: ", lens.std())
    print("mean: ", lens.mean())
    singletons = np.array([counts[c] for c in counts if counts[c] == 1]).sum()
    print("singletons: ", singletons)
    return clustering


def get_clustering(similarities, n_clusters=100, algorithm=None):
    if not algorithm:
        algorithm = "agglomerative"
    clustering = None
    if algorithm == "agglomerative":
        clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity='precomputed', linkage='complete')
    elif algorithm == "dbscan":
        clustering = DBSCAN(metric='precomputed')
    else:
        print("unrecognized algorithm", algorithm)
        print("please choose one of: agglomerative, dbscan")
    u = clustering.fit_predict(similarities)
    print(sklearn.metrics.silhouette_score(similarities, clustering.labels_))
    counts = Counter(u)
    lens = np.array([counts[c] for c in counts])
    print("max size: ", max(lens))
    print("std: ", lens.std())
    print("mean: ", lens.mean())
    singletons = np.array([counts[c] for c in counts if counts[c] == 1]).sum()
    print("singletons: ", singletons)
    return clustering


def create_clusters_from_clustering(clustering, df):
    labs = clustering.labels_
    clusters = {}
    order = sorted(list(zip(df.id, df.rhetorical_sequence)), key=sort_indexes)
    for i in range(len(order)):
        if labs[i] not in clusters.keys():
            clusters[labs[i]] = {'gesture_ids': [],
                                 'id': labs[i]}
        clusters[labs[i]]['gesture_ids'].append(order[i][0])
    return clusters


# TODO delete this and only go through the rhetorical clusterer as progress_apply is better.
# TODO also delete clusters with only one gesture
def cluster_fixed_tag(df):
    ftc = {
        -1: {
            'gesture_ids': []
        }
    }

    for i, row in df.iterrows():
        ru = row['rhetorical_units']
        if isinstance(row['rhetorical_units'], dict):
            ru = ru['sequence']
        elif isinstance(row['rhetorical_units'], list):
            ru = " ".join([el['sequence'] for el in ru])
        else:
            ftc[-1]['gesture_ids'].append(row['id'])
            continue
        if ru in ftc.keys():
            ftc[ru]['gesture_ids'].append(row['id'])
        else:
            ftc[ru] = {
                'gesture_ids': [row['id']]
            }

    return ftc


# average, max, sd cluster size
def get_cluster_metrics(clusters):
    lengths = []
    for c in clusters.keys():
        lengths.append(len(clusters[c]['gesture_ids']))
    lengths = np.array(lengths)
    print("average cluster size:", lengths.mean())
    print("std cluster size:", lengths.std())
    print("max cluster size:", lengths.max())
    without_max = np.delete(lengths, np.argwhere(lengths == lengths.max()))
    print("average without_max size:", without_max.mean())
    print("std without_max size:", without_max.std())
    print("max without_max size:", without_max.max())


def count_gestures_in_clusters(clusters):
    total = []
    for c in clusters.keys():
        total += clusters[c]['gesture_ids']
    return len(total)


class RhetoricalClusterer:
    def __init__(self, df):
        self.bucket = "parsed_transcript_bucket"
        self.df = df
        self.clusters = {}
        self.c_id = 0
        self.total_clusters_created = 0
        self.similarities = []
        self.clustering = None
        return

    def initialize_clusterer(self, df=None):
        self.df['rhetorical_sequence'] = ''
        self.df['rhetorical_units'] = None

        self.df = df if df else self.df
        files = list(set(list(self.df['video_fn'])))
        for f in tqdm(files):
            gesture_ids = self.df.loc[self.df['video_fn'] == f]['id'].tolist()
            self.get_all_encodings_for_video_fn(f, gesture_ids)
        print("could not get", len(self.df[self.df['rhetorical_sequence'] == '']), "out of ", len(self.df), "gestures")
        return

    def get_all_encodings_for_video_fn(self, video_fn, gesture_ids):
        rhetorical_parse_file = multi_replace(video_fn, VID_EXTENSION_REPLACEMENTS)
        # download and delete?
        try:
            f = download_blob(self.bucket, rhetorical_parse_file, "tmp.rhet", should_show=False)
        except:
            print("couldn't get rhetorical parse file for ", rhetorical_parse_file)
            return
        content = get_parse_data("tmp.rhet")
        # os.remove("tmp.rhet")
        en, texts = get_sequence_encoding(content=content)

        for g_id in gesture_ids:
            ind = self.df.index[self.df['id'] == g_id].tolist()
            if len(ind) != 1:
                print("Unmatching number of gestures possible for gesture ID")
                print("ID: ", g_id)
                print("located indexes: ", ind)
                continue
            ind = ind[0]
            transcript_to_match = self.df.iloc[ind]['transcript']
            if transcript_to_match == '':
                continue

            text_range, text_chunks = get_matching_words(transcript_to_match, texts)
            if text_range:
                sequence = en[text_range[0]:text_range[-1]+1]
                rhetorical_units = []
                for i in range(len(text_chunks)):
                    rhetorical_units.append({
                        'text': text_chunks[i],
                        'sequence': en[text_range[i]]
                    })
                self.df.at[ind, 'rhetorical_units'] = rhetorical_units
                self.df.at[ind, 'rhetorical_sequence'] = sequence
            else:
                #print("Could not match text to video fn: ", video_fn)
                #print("Gesture ID: ", g_id)
                #print("Index: ", ind)
                #print("Transcript: ", transcript_to_match)
                continue

    def get_levenshtein_similarities(self, df=None):
        if df is None:
            df = self.df
        print("getting edit distances")
        words = []
        order = list(zip(df.id, df.rhetorical_sequence))  # keep dict in order to sort and
        for k, v in sorted(order, key=sort_indexes):  # assign proper distances to it.
            words.append(" ".join(v))
        lev_similarity = []
        for i in tqdm(range(len(words))):
            w = words[i]
            lev_similarity.append([edlib.align(w, w2)['editDistance'] for w2 in words])
        self.similarities = np.array(lev_similarity)
        return self.similarities

    def cluster_sequences(self, n_clusters=100, df=None):
        if df is None:
            df = self.df
        if self.clusters and len(self.clusters) == n_clusters:
            print("already have rhetorical clusters")
            return self.clusters
        if 'rhetorical_sequence' not in list(df):
            print("getting initial sequences")
            self.initialize_clusterer()
        if not len(self.similarities):
            self.get_levenshtein_similarities(df)

        print("getting clustering")
        self.clustering = get_clustering(self.similarities, n_clusters, algorithm="agglomerative")
        u = self.clustering.fit_predict(self.similarities)
        clusters = create_clusters_from_clustering(self.clustering, df)
        self.clusters = clusters
        return clusters

    # ONLY DO THIS if the dataset has been spliced by rhetorical units.
    def cluster_by_units(self, n_clusters=100, df=None, algorithm="agglomerative"):
        if df is None:
            df = self.df
        if 'rhetorical_units' not in list(self.df):
            print("DONT USE THIS IF YOU HAVENT SPLICED BY RHETORICAL UNITS, YOU FOOL")
            return
        similarities = get_lev_similarities_by_rhetorical_units(df)
        clustering = get_clustering(similarities, n_clusters=n_clusters, algorithm=algorithm)
        return create_clusters_from_clustering(clustering, df)

    def get_sentences_for_cluster(self, cluster_id):
        c = self.clusters[cluster_id]
        rows = self.df[self.df['id'].isin(c['gesture_ids'])]
        return list(rows['transcript'])

    def get_clusters_with_sizes(self):
        return [{c: len(self.clusters[c]['gesture_ids'])} for c in self.clusters.keys()]

    def get_average_gesture_id_from_cluster(self, cluster_id):
        c = self.clusters[cluster_id]
        minimum_gesture_id = 0
        min_diff = 10000
        for i in range(len(c['gesture_ids'])):
            if c['similarities'][i].mean() < min_diff:
                min_diff = c['similarities'][i].mean()
                minimum_gesture_id = c['gesture_ids'][i]
        return minimum_gesture_id

    def get_silhouette_score(self):
        return sklearn.metrics.silhouette_score(self.similarities, self.clustering.labels_)

    def try_silhouette_scores(self, similarities=None, n_clusters=None, algo='agglomerative'):
        if similarities is None:
            similarities = self.similarities
        if n_clusters is None:
            n_clusters = [15, 40, 60, 80, 100, 150, 200]
        for n in n_clusters:
            print("trying ", n)
            clustering = AgglomerativeClustering(n_clusters=n, affinity='precomputed', linkage='average')
            # similarities is nxn matrix (lev_sim)
            u = clustering.fit_predict(similarities)
            print(sklearn.metrics.silhouette_score(similarities, clustering.labels_))
            counts = Counter(u)
            lens = np.array([counts[c] for c in counts])
            print("max size: ", max(lens))
            print("std: ", lens.std())
            print("mean: ", lens.mean())
            singletons = np.array([counts[c] for c in counts if counts[c] == 1]).sum()
            print("singletons: ", singletons)

    def plot_silhouette_scores(self, similarities=None, n_clusters=None):
        if similarities is None:
            similarities = self.similarities
        if n_clusters is None:
            n_clusters = [15, 50, 80, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
        scores = []
        for n in n_clusters:
            print("trying ", n)
            clustering = AgglomerativeClustering(n_clusters=n, affinity='precomputed', linkage='complete')
            u = clustering.fit_predict(similarities)
            scores.append(sklearn.metrics.silhouette_score(similarities, clustering.labels_))

        plt.plot(n_clusters, scores)


    #clusters_rhet_dbscan_large
    def cluster_to_fixed_tag(self, row):
        print(row)
        ru = row['rhetorical_units']
        if isinstance(row['rhetorical_units'], dict):
            ru = ru['sequence']
        elif isinstance(row['rhetorical_units'], list):
            ru = " ".join([el['sequence'] for el in ru])
        else:
            self.clusters[-1]['gesture_ids'].append(row['id'])
            return
        if ru in self.clusters.keys():
            self.clusters[ru]['gesture_ids'].append(row['id'])
        else:
            self.clusters[ru] = {
                'gesture_ids': []
            }


    def cluster_fixed_tag(self, df):
        self.clusters = {
            -1: {
                'gesture_ids': []
            }
        }

        df.progress_apply(self.cluster_to_fixed_tag)
        return self.clusters
