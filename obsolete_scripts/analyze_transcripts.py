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
