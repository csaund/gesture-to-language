#!/usr/bin/env python
import os
import argparse
import io
import subprocess
import json
import copy

from sets import Set

print "loading modules"
import nltk
from nltk.corpus import wordnet as wn
from nltk import sentiment as sent
print "initializing analyzer"
Analyzer = sent.vader.SentimentIntensityAnalyzer()

# for sentiment?
import pandas as pd

POS_I_LIKE = ["VBN", "VB", "VBG", "NN", "RB", "PRP$"]

stop_words = list(set(nltk.corpus.stopwords.words('english')))

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


def get_transcript_structure(gesture_transcripts):
    for phrase in gesture_transcripts:
        phrase_transcript = phrase["phase"]["transcript"]
        print "analyzing phrase: " + phrase_transcript
        p_tokens = nltk.word_tokenize(phrase_transcript)
        phrase["phase"]["tokens"] = p_tokens
        p_structure = nltk.pos_tag(p_tokens)
        phrase["phase"]["structure"] = p_structure
        phrase_sentiment = Analyzer.polarity_scores(phrase_transcript)
        phrase["sentiment"] = phrase_sentiment
        token_index = 0

        for gesture in phrase["gestures"]:
            gesture_transcript = gesture["transcript"]
            g_tokens = nltk.word_tokenize(gesture_transcript)
            g_structure = []
            gesture["hypernyms"] = {}
            sentiment = Analyzer.polarity_scores(gesture_transcript)
            gesture["sentiment"] = sentiment

            structure_index = 0
            for s_index in range(len(g_tokens)):
                pos_struct = p_structure[token_index]
                g_structure.append(pos_struct)
                token_index += 1

                if(pos_struct[1] in POS_I_LIKE):
                    gesture["hypernyms"][pos_struct[1]] = get_hypernyms(pos_struct[0])


            gesture["structure"] = g_structure


    return gesture_transcripts


def read_file(filepath):
    with open(filepath) as f:
        gesture_transcripts = json.load(f)
    return gesture_transcripts

def write_data(fp, data):
    with open(fp, 'w') as f:
        json.dump(data, f, indent=4)
    f.close()


# give this megyn-kelly.mp4 too
if __name__ == '__main__':
    print "running program"
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'path', help='Long mp4 file to be segmented into gestures')
    args = parser.parse_args()

    vid_path = args.path
    filename_base = vid_path.split('/')[-1].split('.')[-2]
    json_fp = filename_base + '/' + filename_base + '_transcripts.json'
    json_outfp = filename_base + '/' + filename_base + '_transcripts_analyzed.json'

    gesture_transcripts = read_file(json_fp)
    analyzed = get_transcript_structure(gesture_transcripts)
    write_data(json_outfp, analyzed)
