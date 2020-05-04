#!/usr/bin/env pythons
import json
import os
import numpy as np
import random
import time
from tqdm import tqdm
from sklearn.neighbors.nearest_centroid import NearestCentroid
from GestureMovementHelpers import *
from common_helpers import *
import copy

# g returned must look like
# {
#   'phase':
#       {
#           'video_fn': '1._Course_introduction-p2J7wSuFRl8.webm',
#           'end_seconds': 401.034368,
#           'transcript': "thing I need to explain is this there's roughly speaking two ways",
#           'start_seconds': 399.232566
#       },
#  'gestures':
#       [
#           {'end_seconds': 401.034368,
#            'start_seconds': 399.232566
#            }
#       ],
#   'speaker': 'shelly',
#   'id': 8,
#   'words': [
#           {
#               'word': 'thing',
#               'word_end': 398,
#               'word_start': 398
#            },
#            {
#               'word': 'I',
#               'word_end': 398,
#               'word_start': 398
#            },
#            ...
#         ],
#   'keyframes':
#               [
#                   {
#                       'y': [127, 132, 196, 239, 120, 171, 215, 71, 62, 60, 229, 223, 221, 217, 215, 236, 234, 232, 228, 242, 238, 235, 233, 243, 241, 238, 235, 243, 241, 239, 237, 252, 248, 248, 248, 248, 263, 261, 256, 252, 266, 262, 254, 251, 267, 261, 254, 253, 265, 261, 257, 255],
#                       'x': [338, 293, 276, 266, 383, 412, 426, 344, 330, 351, 427, 426, 436, 443, 451, 432, 424, 417, 412, 428, 419, 412, 407, 423, 417, 410, 405, 419, 414, 410, 406, 263, 257, 248, 239, 231, 253, 254, 258, 261, 260, 263, 265, 266, 265, 269, 269, 269, 270, 274, 272, 272]
#                    },
#                    ...
#               ]
#


def get_time_split_by_frame(g, f):
    end = g['phase']['transcript']['end_seconds']
    start = g['phase']['transcript']['start_seconds']
    frame_second = start + f * ((end-start) / len(g['keyframes']))
    return frame_second


# only going to be rough approximation because we only have the gesture start
# and end time, and start and end
def get_words_split_by_time(g, t):
    g1_words = [w['word'] for w in g['words'] if w['word_start'] < t]
    g2_words = [w['word'] for w in g['words'] if w['word_start'] >= t]
    return g1_words ,g2_words


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
        'phase': {
            'video_fn': gesture['phase']['video_fn'],
            'start_seconds': 0,
            'end_seconds': 0,
            'transcript': ''
        },
        'speaker': gesture['speaker'],
        'id': gesture['id'],
        'words': [],
        'keyframes': []
    }
    g1 = copy.deepcopy(template)
    g2 = copy.deepcopy(template)
    time_split = get_time_split_by_frame(gesture, frame)
    g1['words'], g2['words'] = get_words_split_by_time(gesture, time_split)
    g1['phase']['transcript'] = [w['word'] for w in g1['words']]
    g2['phase']['transcript'] = [w['word'] for w in g2['words']]
    g1['keyframes'], g2['keyframes'] = get_motion_split_by_frame(gesture, frame)
    g1['phase']['start_seconds'] = gesture['phase']['start_seconds']
    g1['phase']['end_seconds'] = gesture['phase']['start_seconds'] + time_split
    g2['phase']['start_seconds'] = gesture['phase']['start_seconds'] + time_split
    g1['phase']['end_seconds'] = gesture['phase']['end_seconds']
    g2['id'] = str(gesture['id']) + "-" + str(frame)
    return g1, g2


# detect lack of movement
# return the frame at which we should splice the gesture if there is one
# is there isn't one, return None
def detect_splice_frame(gesture):
    f = get_low_motion_frame(gesture['keyframes'])
    if gesture['id']:
        return 0
    return None


class GestureSplicer():
    def __init__(self, agd):
        # this gesture data needs to be full and complete, and include the transcript.
        # need to update GSM to have agd be ALL data, including transcript.
        # time to convert to dfs.
        self.agd = agd

    def splice_gestures(self):
        new_gesture_data = []
        for g in self.agd:
            new_gesture_data.append(self.splice_gesture(g))
        return

    def splice_gesture(self, gesture):
        # detect lack of movement by finding period of high movement,
        # then period of low movement
        # splice right when high movement ends?
        frame = detect_splice_frame(gesture)
        if frame:
            g1, g2 = splice_gesture_at_frame(gesture, frame)
            self.splice_gesture(g2)
            return g1
        else:
            return gesture

    # from motion, detect where is a good place to splice the gesture, if any.
    # importantly, only returns FIRST place this should happen.

    # actually perform the splicing


