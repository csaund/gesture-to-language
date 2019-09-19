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
# parser = nltk.parse.malt.MaltParser()

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
        phrase_transcript = phrase["phase"]["transcript"]
        p_tokens = nltk.word_tokenize(phrase_transcript)
        phrase["phase"]["tokens"] = p_tokens
        p_structure = nltk.pos_tag(p_tokens)
        phrase["phase"]["structure"] = p_structure
        token_index = 0
        for gesture in phrase["gestures"]:
            gesture_transcript = gesture["transcript"]
            g_tokens = nltk.word_tokenize(gesture_transcript)
            gesture["tokens"] = g_tokens
            g_structure = []
            structure_index = 0
            while g_tokens[structure_index] == p_tokens[token_index]:
                g_structure.append(p_structure[token_index])
                structure_index += 1
                token_index += 1
                if(token_index >= len(p_tokens)):
                    break
                elif(structure_index >= len(g_tokens)):
                    break
            gesture["structure"] = g_structure


        # trans_index = 0
        # for gest_index in range(0, len(phrase["gestures"])):
        #     while gestures[gest_index]["transcript"][trans_index] == phrase["transcript"][trans_index]:
        #         trans_index++;

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
