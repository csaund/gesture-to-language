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


def flatten(li): return flatten(li[0]) + (flatten(li[1:]) if len(li) > 1 else []) if type(li) is list else [li]


def multi_index(li, val): return [i for i in range(len(li)) if li[i] == val]


# https://stackoverflow.com/questions/15175142/how-can-i-do-multiple-substitutions-using-regex-in-python
def multi_replace(text, replacement_dict=None):
    if replacement_dict is None:
        replacement_dict = PUNCTUATION_REPLACEMENTS
    regex = re.compile("(%s)" % "|".join(map(re.escape, replacement_dict.keys())))
    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda mo: replacement_dict[mo.string[mo.start():mo.end()]], text)


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
    def __init__(self, gestures):
        self.bucket = "parsed_transcript_bucket"
        self.agd = {}
        self.clusters = {}
        self.c_id = 0
        self.total_clusters_created = 0
        files = list(set([g['phase']['video_fn'] for g in gestures]))
        for f in files:
            self.get_all_encodings_for_video_fn(f, gestures)
        print("got", len(self.agd), "out of ", len(gestures), "gestures")
        return

    def get_all_encodings_for_video_fn(self, video_fn, gestures):
        gs = [g for g in gestures if g['phase']['video_fn'] == video_fn]
        rhetorical_parse_file = multi_replace(video_fn, VID_EXTENSION_REPLACEMENTS)
        # download and delete?
        try:
            f = download_blob(self.bucket, rhetorical_parse_file, "tmp.rhet")
        except:
            print("couldn't get rhetorical parse file for ", rhetorical_parse_file)
        content = get_parse_data("tmp.rhet")
        os.remove("tmp.rhet")
        en, texts = get_sequence_encoding(content=content)
        i = 0
        for g in gs:
            print(i, "/", len(gs), "\r", end="")
            i += 1
            transcript_to_match = g['phase']['transcript']
            text_range = get_matching_words(transcript_to_match, texts)
            if text_range:
                k = g['id']
                transcript = g['phase']['transcript']
                sequence = en[text_range[0]:text_range[-1]+1]
                self.agd[k] = {
                                'id': k,
                                'transcript': transcript,
                                'sequence': sequence
                               }

    def get_encoding_for_gesture(self, g):
        transcript_to_match = g['phase']['transcript']
        rhetorical_parse_file = multi_replace(g['phase']['video_fn'], VID_EXTENSION_REPLACEMENTS)
        # download and delete?
        try:
            f = download_blob(self.bucket, rhetorical_parse_file, "tmp.rhet")
        except:
            print("couldn't get rhetorical parse for ", rhetorical_parse_file)
        content = get_parse_data("tmp.rhet")
        os.remove("tmp.rhet")
        en, texts = get_sequence_encoding(content=content)
        text_range = get_matching_words(transcript_to_match, texts)
        if not text_range:
            print("No rhetorical encoding found for gesture ", g['id'])
            return None
        return en[text_range[0]:text_range[-1] + 1]

    def cluster_sequences(self):
        # https://gist.github.com/codehacken/8b9316e025beeabb082dda4d0654a6fa
        words = []
        for k, v in sorted(self.agd.items()):
            words.append(" ".join(v['sequence']))

        lev_similarity = []
        print("getting edit distances")
        for i in range(len(words)):
            print('\r', i, end='                 ')
            w = words[i]
            lev_similarity.append([edlib.align(w, w2)['editDistance'] for w2 in words])

        lev_similarity = np.array(lev_similarity)

        print("getting agglomerative clustering")
        agg = AgglomerativeClustering(n_clusters=100, affinity='precomputed', linkage='complete')
        u = agg.fit_predict(lev_similarity)

        i = 0
        for k, v in sorted(self.agd.items()):
            self.agd[k]['cluster_id'] = u[i]
            i += 1

        return(agg, u, lev_similarity)


def get_sentences_for_rhetorical_cluster(GSM, c_id):
    Rh = GSM.RhetoricalClusterer
    t_ids = [Rh.agd[k]['id'] for k in Rh.agd.keys() if Rh.agd[k]['cluster_id'] == c_id]
    transcripts = [GSM.get_gesture_transcript_by_id(i) for i in t_ids]
    for t in transcripts:
        print(t['phase']['transcript'])