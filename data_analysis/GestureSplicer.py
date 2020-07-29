#!/usr/bin/env pythons
from GestureMovementHelpers import get_first_low_motion_frame
from common_helpers import *
from data_analysis.VideoManager import VideoManager
from data_analysis.RhetoricalClusterer import multi_replace
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm
import re

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
        #print(g['id'])
        #print(g['end_seconds'])
        #print(f)
        #print('0 length keyframes here')
        return end
    frame_second = start + f * ((end-start) / len(g['keyframes']))
    return frame_second


def get_frame_split_by_time(g, t):
    #print('getting frame split by time')
    #print('num keyframes: ', len(g['keyframes']))
    #print('start s:', g['start_seconds'])
    start = g['start_seconds']
    end = g['end_seconds']
    n_frames = len(g['keyframes'])
    fps = n_frames / (end - start)
    frame = fps * (t - start)
    #print('start s: ', start)
    #print('calculated fps: ', fps)
    #print('t to get: ', t)
    #print('calculated frame: ', frame)
    if frame < MINIMUM_FRAME_LENGTH:
        if g['id'] ==25593:
            print('start', g['start_seconds'])
            print('end', g['end_seconds'])
            print('nframes ', n_frames)
            print('fps', fps)
            print(frame)
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
#def get_rhetorical_splice_times_for_gesture(gesture):
    # print("DOING THIS FOR GESTURE ID", gesture['id'])
#    transcript = gesture['transcript']
 #   units = gesture['rhetorical_units']
 #   words = gesture['words']
 #   transcript = multi_replace(transcript, replacement_dict=PARSER_REPLACEMENTS)
  #  words = flatten([split_apostrophes(w['word']) for w in words])

 #   if not units:
  #      # no rhetorical units are found
 #       return None, None

 #   j = 0
#    ends = []
 #   if not isinstance(units, list):     # the rhetorical units only contain 1, so we can't parse it further
 #       return None, None
 #   for u in units:
 #       text = u['text'].split(" ")
  #      i = get_first_occurrence_of_word(text, words[j])
  #      if not i:
            # print('no match found for transcript')
 #           return None, None
 #       else:
 #           while i < len(text) and j < len(words) and text[i] == words[j]:
  #              i += 1
  #              j += 1
   #         word_index = get_original_word_index(gesture['words'], words, j)
  #          if word_index >= len(gesture['words']):
 #               break
 #           ends.append(get_next_word_end(gesture['words'], word_index-1))
  #  return ends, units


def splice_gesture_by_rhetorical_parses(g):
    #print("trying to splice")
    times, units = temp_get_rhetorical_splice_times_for_gesture(g)

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
        new_g = temp_splice_gesture_by_rhetorical_parses(row)
        if len(new_g) >= 2:
            new_gestures += new_g
    return new_gestures


def splice_all_gestures_by_rhetorical_parses(df):
    if 'rhetorical_units' not in list(df):
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

    def splice_gestures(self, df, splice_type=None):
        # this hurts real bad but we have to do it like this. we have to iterate over every one
        # we can't use apply because we're adding and deleting rows from the df.
        print("splicing gestures")
        if splice_type is None:
            print("missing required parameter splice_type in gesture splicer. Must be 'motion' or 'rhetorical'")
            return df
        if splice_type == 'motion':
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
            return additional.reset_index(inplace=True, drop=True)
        elif splice_type == 'rhetorical':
            return splice_all_gestures_by_rhetorical_parses(df)
        else:
            print('unrecognized splice type', splice_type)
            return df

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




def get_all_occurrences_of_word(text, word):
    occ = []
    for i in range(len(text)):
        if text[i] == word:
            occ.append(i)
    return occ

def split_remove_blanks(text):
    l = text.split(' ')
    while "" in l:
        l.remove("")
    return l

# sequence to match is A, B, C, D, E, F
# given to text blAhblAhblAh A, B, C, F, A, B, C,D, E, F
# find occurrence of A
# if B the thing immeditely following it?
# is C following that?
# etc...
# when chain is broken, find next occurrence of A
# ...

# while text_matches_A:
#   while next_thing_matches:
#       keep_checking
#       if end_of_sequence
#           return


# todo come back to this
def temp_get_rhetorical_splice_times_for_gesture(gesture):
    #print("DOING THIS FOR GESTURE ID", gesture['id'])
    transcript = gesture['transcript']
    units = gesture['rhetorical_units']
    words = gesture['words']
    transcript = multi_replace(transcript, replacement_dict=PARSER_REPLACEMENTS)
    words = flatten([split_apostrophes(w['word']) for w in words])

    if not units:
        # no rhetorical units are found
        return None, None

    j = 0
    ends = []
    if not isinstance(units, list):     # the rhetorical units only contain 1, so we can't parse it further
        return None, None
    elif len(units) == 1:
        return None, None
    # find occurrence of A
    # see if next thing matches
    # if next thing doesn't match, go to next occurrence of A
    # if next thing does match, see if next thing matches

    orig_j = 0
    for u in units:
        #print('NEXT UNIT!')
        text = split_remove_blanks(u['text'])
        j = orig_j
        # while we're still in this unit
        i = 0
        while i < len(text) and j < len(gesture['words']):
            #print('ON TEXT: ', text, 'id:', gesture['id'])
            #print('comparing', text[i], 'with', words[j])
            if text[i] == re.sub("[\.\$\-\:]", '', words[j]):  # if the words match, continue
                i += 1
                j += 1
            else:       # if not, start over to find the next word match
                #print('resetting j to ', orig_j)
                #print('while i is', i)
                j = orig_j
                # if we happen to be at the next start poing
                if text[i] == re.sub("[\.\$\-\:]", '', words[j]):
                    if i+1 < len(text) and j+1 < len(words) and text[i+1] == re.sub("[\.\$\-\:]", '', words[j+1]):
                        continue
                    else:
                        #print("else1")
                        i += 1
                else:
                    #print("else2")
                    i += 1

            # at the end of the unit, we've found an end word
        #print('js new reset is ', orig_j)
        #print('appending end word: ', gesture['words'][j-3])
        orig_j = j
        ends.append((gesture['words'][j - 3]['word_start']))

    #print('GESTURE START TIME: ', gesture['start_seconds'])
    #print('GESTURE END TIME: ', gesture['end_seconds'])
    #print('WORD TIMINGS: ', ends)
    return ends, units

# todo delete this immediately
def temp_splice(df):
    if 'rhetorical_units' not in list(df):
        print('need rhetorical parses to perform parse.')
        print('please initialize rhetorical clusterer before parsing.')
        return df

    new_gestures = temp_get_new_gestures_by_rhetorical_parses(df)

    print("rebuilding dataframe")
    to_del = [g['id'] for g in new_gestures]
    print("number of new gestures created: ", len(new_gestures))
    ng_series = [pd.Series(g) for g in new_gestures]
    short_df = df.drop(df.index[df['id'].isin(to_del)])
    additional = short_df.append(ng_series)
    return additional.reset_index(drop=True)

def temp_get_new_gestures_by_rhetorical_parses(df):
    new_gestures = []
    for index, row in tqdm(df.iterrows()):
        # print(row)
        new_g = temp_splice_gesture_by_rhetorical_parses(row)
        if len(new_g) >= 2:
            new_gestures += new_g
    return new_gestures


# TODO update this still more because it doesn't catch when you splice something
# but leave multiple rhetorical parses, which happens when things are quite short
def temp_splice_gesture_by_rhetorical_parses(g):
    #print("trying to splice")
    times, units = temp_get_rhetorical_splice_times_for_gesture(g)

    if g['id'] == 25593:
        print("FOLLOW THIS GESTURE")
        print('times: ', times)
        print('units: ', units)

    if not times:
        if g['id']==25593:
            print("got no times for our gesture??")
        return [g]

    frames = np.array([get_frame_split_by_time(g, t) for t in times])
    if g['id']==25593:
        print('frames to split: ', frames)
    gs = []
    gest = g
    for i in range(len(frames)):
        f = frames[i]
        if not f:
            continue
        g1, g2 = temp_splice_gesture_at_frame(gest, f)
        # TODO clean this up so this doesn't happen here
        g1['rhetorical_units'] = units[i]
        gs.append(g1)
        gest = g2
        frames = frames - f
        if i == len(frames)-1:
            g2['rhetorical_units'] = units[-1]
            gs.append(g2)

    if g['id'] == 25593:
        print('UPDATED')
        print('returning gs', len(gs))
        for g in gs:
            print(g['rhetorical_units'])
    return gs

def temp_get_time_split_by_frame(g, f):
    end = g['end_seconds']
    start = g['start_seconds']
    if not len(g['keyframes']):
        #print('0 length keyframes here')
        #print('id:', g['id'])
        #print('end seconds:', g['end_seconds'])
        #print('frame: ', f)
        return end
    frame_second = start + f * ((end-start) / len(g['keyframes']))
    return frame_second


def temp_splice_gesture_at_frame(gesture, frame):
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
    time_split = temp_get_time_split_by_frame(gesture, frame)
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