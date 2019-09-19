#!/usr/bin/env python
import os
import argparse
import io
import subprocess
import json
import copy

from sets import Set
import nltk
from nltk.corpus import wordnet as wn

# for sentiment?
import pandas as pd

# for sentence structure?
from stat_parser import Parser

##
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
       #print w
       return w

def get_hypernyms(w0):
    syn = wn.synsets(morph(w0))[0]
    return syn.hypernyms()

# print wnexpand(('big'))
# print
# print get_hypernyms('big')

def get_wn_forms(gesture_transcripts):
    for phrase in gesture_transcripts:
        for gesture in phrase["gestures"]:
            print gesture
            # do pos tagging
            # get subject, do hypernyms?
            transcript = gesture["transcript"]
            tokens = nltk.word_tokenize(transcript)
            gesture["tokens"] = tokens
            # I don't think it makes sense to take out stopwords --
            # they often carry very important sentence structure information
            # and are extremely relevant to the gesture
            # i.e. ("all of you", "only mine is down")
            # stopped = [w for w in tokens if not w in stop_words]
            gesture["structure"] = Parser.parse(transcript)

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
    analyzed = get_wn_forms(gesture_transcripts)
    write_data(json_outfp, analyzed)
