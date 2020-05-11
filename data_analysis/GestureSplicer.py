#!/usr/bin/env pythons
from GestureMovementHelpers import get_first_low_motion_frame
from common_helpers import *
from data_analysis.VideoManager import VideoManager
import copy
import pandas as pd
from tqdm import tqdm


def get_time_split_by_frame(g, f):
    end = g['end_seconds']
    start = g['start_seconds']
    frame_second = start + f * ((end-start) / len(g['keyframes']))
    return frame_second


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
        'keyframes': []
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
    g1['end_seconds'] = gesture['start_seconds'] + time_split
    g2['start_seconds'] = gesture['start_seconds'] + time_split
    g1['end_seconds'] = gesture['end_seconds']
    g2['id'] = str(gesture['id']) + "." + str(frame)
    return g1, g2


def get_first_occurrence_of_word(text, word):
    for i in range(len(text)):
        if text[i] == word:
            return i
    print("could not find word in text: ", word)
    print(text)
    return None


class GestureSplicer():
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

    def splice_gesture_by_rhetorical_parses(self, df=None):
        df = df if df else self.df
        if ('motion_feature_vec' not in list(df)) or ('rhetorical_units' not in list(df)):
            print('need rhetorical parses to perform parse.')
            print('please initialize rhetorical clusterer before parsing.')
            return df

        # get start and end of first rhetorical phrase
        g = df.iloc[0]
        words = g['words']
        units = g['rhetorical_units']

        j = 0
        starts = []
        for u in units:
            text = u['text'].split(" ")
            i = get_first_occurrence_of_word(text, words[j]['word'])
            print("adding start", words[j])
            starts.append(words[j]['word_start'])
            while text[i] == words[j]['word']:
                i += 1
                j += 1
                if i >= len(text) or j >= len(words):
                    break
                if len(words[j]['word'].split("'")) >= 2:
                    if words[j]['word'].split("'")[0] == text[i]:
                        i += 2
                        j += 1
            continue
        return starts

    # given an array of text, find the first index at which the word occurs




