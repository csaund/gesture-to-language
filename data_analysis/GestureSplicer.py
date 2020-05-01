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



# Get gestures given to it
# Goes through and looks for pauses in motion
# Splices the gesture to create two new gestures from before and after the motion will full features
# continues on to the end of the gesture (can splice one gesture multiple times)
class GestureSplicer():
    def __init__(self, agd):
        self.agd = agd

    def splice_gesture(self, gesture):
        # detect lack of movement
        frame = self.detect_splice_frame(gesture)
        if frame:
            g1, g2 = self.splice_gesture_at_frame(gesture, frame)
            self.splice_gesture(g2)
        else:
            return gesture

    # from motion, detect where is a good place to splice the gesture, if any.
    # importantly, only returns FIRST place this should happen.
    def detect_splice_frame(self, gesture):
        # detect lack of movement
        return 0

    # actually perform the splicing
    def splice_gesture_at_frame(self, gesture, frame):
        g1 = {}
        g2 = {}
        return g1, g2