#!/usr/bin/env pythons
from GestureMovementHelpers import get_first_low_motion_frame
from common_helpers import *
from data_analysis.VideoManager import VideoManager
from data_analysis.RhetoricalClusterer import multi_replace
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm

MINIMUM_FRAME_LENGTH = 20


PARSER_REPLACEMENTS = {
    "n't": " nt",
    "'s": " s",
    "'m": " m",
    "'re": " re",
    "'ve": " ve",
    "'ll": " ll",
    "'": " ",
}


def get_time_split_by_frame(g, f):
    end = g['end_seconds']
    start = g['start_seconds']
    if not len(g['keyframes']):
        print('0 length keyframes here')
        return end
    frame_second = start + f * ((end-start) / len(g['keyframes']))
    return frame_second


def get_frame_split_by_time(g, t):
    start = g['start_seconds']
    end = g['end_seconds']
    n_frames = len(g['keyframes'])
    fps = n_frames / (end - start)
    frame = fps * (t - start)
    if frame < MINIMUM_FRAME_LENGTH:
        return 0
    else:
        return int(frame)


# only going to be rough approximation because we only have the gesture start
# and end time, and start and end
def get_words_split_by_time(g, t):
    g1_words = [w for w in g['words'] if w['word_start'] < t]
    g2_words = [w for w in g['words'] if w['word_start'] >= t]
    return g1_words, g2_words


def get_motion_split_by_frame(g, f):
    g1_keys = g['keyframes'][:f]
    g2_keys = g['keyframes'][f:]
    return g1_keys, g2_keys


# Get gestures given to it
# Goes through and looks for pauses in motion
# Splices the gesture to create two new gestures from before and after the motion will full features
# continues on to the end of the gesture (can splice one gesture multiple times)
def splice_gesture_at_frame(gesture, frame):
    template = {
        'video_fn': gesture['video_fn'],
        'start_seconds': 0,
        'end_seconds': 0,
        'transcript': '',
        'speaker': gesture['speaker'],
        'id': gesture['id'],
        'words': [],
        'keyframes': [],
        'rhetorical_sequence': [],
        'rhetorical_units': []
    }
    g1 = copy.deepcopy(template)
    g2 = copy.deepcopy(template)
    time_split = get_time_split_by_frame(gesture, frame)
    g1w, g2w = get_words_split_by_time(gesture, time_split)
    g1['words'] = g1w
    g2['words'] = g2w
    g1['transcript'] = " ".join([w['word'] for w in g1['words']])
    g2['transcript'] = " ".join([w['word'] for w in g2['words']])
    g1k, g2k = get_motion_split_by_frame(gesture, frame)
    g1['keyframes'] = g1k
    g2['keyframes'] = g2k
    g1['start_seconds'] = gesture['start_seconds']
    g1['end_seconds'] = gesture['start_seconds'] + (time_split - gesture['start_seconds'])
    g2['start_seconds'] = gesture['start_seconds'] + (time_split - gesture['start_seconds'])
    g2['end_seconds'] = gesture['end_seconds']
    g2['id'] = str(gesture['id']) + "." + str(frame)
    return g1, g2


# given an array of text, find the first index at which the word occurs
def get_first_occurrence_of_word(text, word):
    for i in range(len(text)):
        if text[i] == word:
            return i
    # print("could not find word in text: ", word)
    # print(text)
    return None


def get_next_word_end(words, i):
    while not i >= len(words):
        if 'word_end' not in words[i].keys():
            i += 1
        else:
            return words[i]['word_end']

    while i >= 0:
        if 'word_end' not in words[i].keys():
            i -= 1
        else:
            return words[i]['word_end']

    print("NO WORD ENDS FOUND IN WORDS", words)
    return None


# given the original list and the new list with updated hooplah, get the adjsuted index.
def get_original_word_index(original_words, new_words, new_index):
    if len(original_words) == len(new_words):
        return new_index
    apostrophe_indexes = [i for i in range(len(original_words)) if original_words[i]['word'].find("'") != -1]
    #print("new index we got was ", new_index)
    #print("original words were: ", original_words)
    #print("think we found apostrophes in indexes: ", apostrophe_indexes)
    for ai in apostrophe_indexes:
        if new_index <= ai:
            return new_index
        else:
            new_index -= 1
    return new_index


def split_apostrophes(w):
    w = multi_replace(w, replacement_dict=PARSER_REPLACEMENTS)
    if len(w.split(" ")) >= 2:
        return w.split(" ")
    return [w]


# TODO fix this. It is "good enough" but the transcript doesn't quite line up
# because lots of words start and end in the same second.
# TODO also currently broken bc of one letter words (terrible parsing issue)
def get_rhetorical_splice_times_for_gesture(gesture):
    # print("DOING THIS FOR GESTURE ID", gesture['id'])
    transcript = gesture['transcript']
    units = gesture['rhetorical_units']
    words = gesture['words']
    #print("pre-transcript: ", transcript)
    transcript = multi_replace(transcript, replacement_dict=PARSER_REPLACEMENTS)
    words = flatten([split_apostrophes(w['word']) for w in words])
    #print("words: ", words)

    if not units:
        # print("NO RHETORICAL UNITS FOUND")
        return None, None

    j = 0
    ends = []
    for u in units:
        #print("new unit!")
        text = u['text'].split(" ")
        #print(text)
        i = get_first_occurrence_of_word(text, words[j])
        if not i:
            # print('no match found for transcript')
            return None, None
        else:
            #print('found at index: ', i)
            while i < len(text) and j < len(words) and text[i] == words[j]:
                #print("comparing ", text[i], "and", words[j])
                i += 1
                j += 1
            word_index = get_original_word_index(gesture['words'], words, j)
            if word_index >= len(gesture['words']):
                break
            #print("word index:", word_index)
            #print("len gesture words: ", len(gesture['words']))
            #print("think we should split at word: ", gesture['words'][word_index-1])
            ends.append(get_next_word_end(gesture['words'], word_index-1))
        # print("looking for this word in that text:", words[j]['word'])
        # i = get_first_occurrence_of_word(text, words[j]['word'])
    return ends, units


def splice_gesture_by_rhetorical_parses(g):
    #print("trying to splice")
    times, units = get_rhetorical_splice_times_for_gesture(g)

    if not times:
        return [g]

    frames = np.array([get_frame_split_by_time(g, t) for t in times])
    gs = []
    gest = g
    for i in range(len(frames)):
        f = frames[i]
        if not f:
            continue
        g1, g2 = splice_gesture_at_frame(gest, f)
        # TODO clean this up so this doesn't happen here
        g1['rhetorical_units'] = units[i]
        gs.append(g1)
        gest = g2
        frames = frames - f
        if i == len(frames)-1:
            g2['rhetorical_units'] = units[-1]
            gs.append(g2)
    return gs


def get_new_gestures_by_rhetorical_parses(df):
    new_gestures = []
    for index, row in tqdm(df.iterrows()):
        # print(row)
        new_g = splice_gesture_by_rhetorical_parses(row)
        if len(new_g) >= 2:
            new_gestures += new_g
    return new_gestures


def splice_all_gestures_by_rhetorical_parses(df):
    if ('motion_feature_vec' not in list(df)) or ('rhetorical_units' not in list(df)):
        print('need rhetorical parses to perform parse.')
        print('please initialize rhetorical clusterer before parsing.')
        return df

    new_gestures = get_new_gestures_by_rhetorical_parses(df)

    print("rebuilding dataframe")
    to_del = [g['id'] for g in new_gestures]
    print("number of new gestures created: ", len(new_gestures))
    ng_series = [pd.Series(g) for g in new_gestures]
    short_df = df.drop(df.index[df['id'].isin(to_del)])
    additional = short_df.append(ng_series)
    return additional.reset_index(drop=True)


class GestureSplicer:
    def __init__(self):
        # this gesture data needs to be full and complete, and include the transcript.
        # need to update GSM to have agd be ALL data, including transcript.
        # time to convert to dfs.
        self.VideoManager = VideoManager()

    def splice_gestures(self, df):
        # this hurts real bad but we have to do it like this. we have to iterate over every one
        # we can't use apply because we're adding and deleting rows from the df.
        print("splicing gestures")
        new_gestures = []
        for index, row in tqdm(df.iterrows()):
            new_g = self._splice_gesture(row)
            if len(new_g) >= 2:
                new_gestures += new_g

        print("rebuilding dataframe")
        to_del = [g['id'] for g in new_gestures]
        ng_series = [pd.Series(g) for g in new_gestures]
        short_df = df.drop(df.index[df['id'].isin(to_del)])
        additional = short_df.append(ng_series)
        return additional.reset_index(inplace=True)

    def _splice_gesture(self, gesture, gestures=None):
        # detect lack of movement by finding period of high movement,
        # then period of low movement
        # splice right when high movement ends?
        if gestures is None:
            gestures = []
        frame = get_first_low_motion_frame(gesture['keyframes'])
        # if audio_features and relative_audio_intensity
        if frame:
            g1, g2 = splice_gesture_at_frame(gesture, frame)
            gestures.append(g1)
            return self._splice_gesture(g2, gestures=gestures)
        else:
            gestures.append(gesture)
            return gestures

    def get_audio_features_by_gesture(self, g):
        af = self.VideoManager.get_audio_features(g['video_fn'], g['start_seconds'], g['end_seconds'])
        return af




