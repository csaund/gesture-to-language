from subprocess import call
import re
import os
from common_helpers import download_blob
from tqdm import tqdm
import string

import numpy as np
from sklearn.cluster import AgglomerativeClustering
import distance
import edlib
import sklearn

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
    for t_index in range(len(texts)):
        t = texts[t_index]
        if t == "":     # skip all blank ones.
            continue
        chunk = t.split(" ")
        cs = []
        for c in chunk:
            cs.append(c.translate(str.maketrans('', '', string.punctuation)))
        # our word isn't in this chunk, start over
        if words[word_counter].translate(str.maketrans('', '', string.punctuation)) not in cs:
            word_counter = 0
            text_indexes = []

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
                        return sorted(list(set(text_indexes)))
                    break
            if not successful_chunk_found:
                word_counter = 0
                text_indexes = []

    return sorted(list(set(text_indexes)))


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
    text_range = get_matching_words(transcript_to_match, texts)
    if not text_range:
        print("No rhetorical encoding found for gesture ", g['id'])
        return None
    return en[text_range[0]:text_range[-1]+1]


class RhetoricalClusterer:
    def __init__(self, df):
        self.bucket = "parsed_transcript_bucket"
        self.df = df
        self.df['rhetorical_sequence'] = ''
        self.clusters = {}
        self.c_id = 0
        self.total_clusters_created = 0
        self.similarities = []
        self.clustering = None
        return

    def initialize_clusterer(self, df=None):
        self.df = df if df else self.df
        files = list(set(list(self.df['video_fn'])))
        for f in files:
            gesture_indexes = self.df.index[self.df['video_fn'] == f].tolist()
            self.get_all_encodings_for_video_fn(f, gesture_indexes)
        print("could not get", len(self.df[self.df['rhetorical_sequence'] == '']), "out of ", len(self.df), "gestures")
        return

    def get_all_encodings_for_video_fn(self, video_fn, gesture_indexes):
        rhetorical_parse_file = multi_replace(video_fn, VID_EXTENSION_REPLACEMENTS)
        # download and delete?
        try:
            f = download_blob(self.bucket, rhetorical_parse_file, "tmp.rhet")
        except:
            print("couldn't get rhetorical parse file for ", rhetorical_parse_file)
        content = get_parse_data("tmp.rhet")
        os.remove("tmp.rhet")
        en, texts = get_sequence_encoding(content=content)

        for g_i in tqdm(gesture_indexes):
            transcript_to_match = self.df.iloc[g_i]['transcript']
            text_range = get_matching_words(transcript_to_match, texts)
            if text_range:
                sequence = en[text_range[0]:text_range[-1]+1]
                self.df.at[g_i, 'rhetorical_sequence'] = sequence

    def cluster_sequences(self):
        if self.clusters:
            print("already have rhetorical clusters")
            return self.clusters

        if 'rhetorical_sequence' not in list(self.df):
            print("getting initial sequences")
            self.initialize_clusterer()

        words = []
        order = list(zip(self.df.id, self.df.rhetorical_sequence))  # keep dict in order to sort and
        for k, v in sorted(order):                                  # assign proper distances to it.
            words.append(" ".join(v))

        lev_similarity = []
        print("getting edit distances")
        for i in range(len(words)):
            print('\r', i, end='                 ')
            w = words[i]
            lev_similarity.append([edlib.align(w, w2)['editDistance'] for w2 in words])

        self.similarities = np.array(lev_similarity)

        print("getting clustering")
        self.clustering = AgglomerativeClustering(n_clusters=100, affinity='precomputed', linkage='complete')
        u = self.clustering.fit_predict(self.similarities)

        labs = self.clustering.labels_
        i = 0
        for k, v in sorted(order):
            if labs[i] not in self.clusters.keys():
                self.make_new_cluster(labs[i])
            ind = self.df.index[self.df['id'] == k].tolist()
            if not ind:
                print("could not find index for gesture ", k)
            self.clusters[labs[i]]['gesture_ids'].append(k)                         # keep all the similarities here to
            self.clusters[labs[i]]['similarities'].append(self.similarities[i])     # calculate centroid of cluster.

        for c in self.clusters.keys():
            self.clusters[c]['centroid_id'] = self.get_average_gesture_id_from_cluster(c)

        return self.clusters

    def make_new_cluster(self, lab):
        self.clusters[lab] = {
            'cluster_id': lab,
            'gesture_ids': [],
            'similarities': [],
            'centroid_id': 0
        }

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

    def try_silhouette_scores(self, similarities=None, n_clusters=None):
        if similarities is None:
            similarities = self.similarities
        if n_clusters is None:
            n_clusters = [15, 40, 60, 80, 100, 150, 200]
        for n in n_clusters:
            print("trying ", n)
            clustering = AgglomerativeClustering(n_clusters=n, affinity='precomputed', linkage='complete')
            # similarities is nxn matrix (lev_sim)
            u = clustering.fit_predict(similarities)
            print(sklearn.metrics.silhouette_score(similarities, clustering.labels_))
            singletons = []
            for lab in list(set(clustering.labels_)):
                l = len([i for i in clustering.labels_ if i == lab])
                if l <= 1:
                    singletons.append(lab)
            print("singletons: ", len(singletons))
