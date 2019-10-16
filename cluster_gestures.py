#!/usr/bin/env pythons
print "importing libs"
import argparse
import pandas as pd
import itertools
import csv
import math
import json
import glob, os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import statistics as stat


# do difference between t1 and t2 for same frame,
# and also difference between p1, p2, p3 etc at same frame
# maybe get r_arm_distance_spread,      l_arm_distance_spread,     r_hand_distance_spread, l_hand_distance_spread
#           [p1_t1 - p2_t1 - p3_t1...], [p1_t1 - p2_t1 - p3_t1...]
#  ds_1 =               [t1                       t1...]
#           [p1_t2 - p2_t2 - p3_t2...], [p1_t2 - p2_t2 - p3_t2...]
#  ds_2 =               [t2                       t2...]
# as represending [p1, p2, p3] etc at t1.
# meanwhile each individual point also has a difference
# [p0-p1], [p1-p2]
#    t1      t2
# p0-1_1 = [p0 - p1]
# p0-1_2 = [p1 - p2]
# so for each t you have
# [ds_1, p0-1_1, p1-1_1, p2-1_1, p3-1_1...]
# need to figure out how to compare subsequences of gesture.

## the above is wayyyyy too complicated for now.
## just get a gesture of high-level features, I think
# All of the following refer to if it happens EVER in the gesture
# [
#   max_vert_accel,
#   max_horiz_accel,
#   l_palm_vert,
#   l_palm_horiz,
#   r_palm_vert,
#   r_palm_horiz,
#   x_oscillate,    # does it go up and down a lot?
#   y_oscillate,
#   hands_together,
#   hands_separate,
#   wrists_up,
#   wrists_down,
#   wrists_outward,
#   wrists_inward,
#   wrists_sweep,
#   wrist_arc,
#   r_hand_rotate,
#   l_hand_rotate,
#   hands_cycle
# ]
## Now since we're only working within a single gesture, don't need to worry about
## Mismatched timings anymore.

# given time sequence x/y: [[p1, p2, p3], [p1', p2', p3']...] calculate maximum velocity between say 5 frames
def get_max_velocity(time_seq):
    return


# given time sequence calculate minimum total x difference between palm points
def get_max_palm_verticalness(time_seq):
    return


# given time sequence calculate minimum total y difference between palm points
def get_max_palm_horizontalness(time_seq):
    return


# returns minimum distance at any frame between point A on right hand and
# point A on left hand.
def min_hands_together(gesture):
    r_hand_keys = get_rl_hand_keypoints(gesture, 'r')
    l_hand_keys = get_rl_hand_keypoints(gesture, 'l')
    min_dist = 1000 # larger than pixel range
    for i in range(0, len(r_hand_keys)-2):
        for j in range(0, len(r_hand_keys[i]['x'])-1):
            r_x = r_hand_keys[i]['x'][j]
            r_y = r_hand_keys[i]['y'][j]
            l_x = l_hand_keys[i]['x'][j]
            l_y = l_hand_keys[i]['y'][j]
            a = np.array((r_x, r_y))
            b = np.array((l_x, l_y))
            dist = np.linalg.norm(a-b)
            min_dist = min(dist, min_dist)
    return min_dist


# returns maximum distance at any frame between point A on right hand and
# point A on left hand
def max_hands_apart(gesture):
    r_hand_keys = get_rl_hand_keypoints(gesture, 'r')
    l_hand_keys = get_rl_hand_keypoints(gesture, 'l')
    max_dist = 0 # larger than pixel range
    for i in range(0, len(r_hand_keys)-2):
        for j in range(0, len(r_hand_keys[i]['x'])-1):
            r_x = r_hand_keys[i]['x'][j]
            r_y = r_hand_keys[i]['y'][j]
            l_x = l_hand_keys[i]['x'][j]
            l_y = l_hand_keys[i]['y'][j]
            a = np.array((r_x, r_y))
            b = np.array((l_x, l_y))
            dist = np.linalg.norm(a-b)
            max_dist = max(dist, max_dist)
    return max_dist

# get maximum "verticalness" aka minimum horizontalness of hands
def palm_vert(gesture, lr):
    hand_keys = get_rl_hand_keypoints(gesture, lr)
    x_min = 1000
    for frame in hand_keys:
        max_frame_dist = max(frame['x']) - min(frame['x'])
        x_min = min(x_min, max_frame_dist)
    return x_min

# get maximum "horizontalness" aka minimum verticalness of hands
def palm_horiz(gesture, lr):
    hand_keys = get_rl_hand_keypoints(gesture, lr)
    y_min = 1000
    for frame in hand_keys:
        max_frame_dist = max(frame['y']) - min(frame['y'])
        y_min = min(y_min, max_frame_dist)
    return y_min


## Total distance from low --> high the wrists move in a single motion
def wrists_up(gesture, lr):
    # use hand avg as wrist proxy for now
    hand_keys = get_rl_hand_keypoints(gesture, lr)
    total_increase = 0
    max_single_increase = 0
    pos = avg(hand_keys[0]['y'])
    going_up = False
    for frame in hand_keys:
        curr_pos = avg(frame['y'])
        # while we're still going up
        if curr_pos >= pos:
            total_increase = total_increase + abs(curr_pos - pos)
            pos = curr_pos
            going_up = True
        # we're lower than we used to be
        else:
            # if we were previously going up
            if going_up:
                max_single_increase = max(max_single_increase, total_increase)
                total_increase = 0
            going_up = False
            pos = curr_pos
    return max_single_increase

## Total distance from high --> low the wrists move
def wrists_down(gesture, lr):
    # use hand avg as wrist proxy for now
    hand_keys = get_rl_hand_keypoints(gesture, lr)
    total_decrease = 0
    max_single_decrease = 0
    pos = avg(hand_keys[0]['y'])
    going_down = False
    for frame in hand_keys:
        curr_pos = avg(frame['y'])
        # while we're still going down
        if curr_pos <= pos:
            total_decrease = total_decrease + abs(curr_pos - pos)
            pos = curr_pos
            going_down = True
        # we're higher than we used to be
        else:
            # if we were previously going down
            if going_down:
                max_single_decrease = max(max_single_decrease, total_decrease)
                total_decrease = 0
            going_down = False
            pos = curr_pos
    return max_single_decrease


## TODO check these bad larries for bugs
def wrists_outward(gesture):
    r_hand = get_rl_hand_keypoints(gesture, 'r')
    l_hand = get_rl_hand_keypoints(gesture, 'l')
    moving_outward = False
    total_outward_dist = 0
    max_outward_dist = 0
    dist = abs(avg(r_hand[0]['x']) - avg(l_hand[0]['x']))
    for i in range(0, len(r_hand)-1):
        curr_dist = abs(avg(r_hand[i]['x']) - avg(l_hand[i]['x']))
        if curr_dist >= dist:
            moving_outward = True
            total_outward_dist = total_outward_dist + abs(curr_dist - dist)
            dist = curr_dist
        else:
            if moving_outward:
                max_outward_dist = max(max_outward_dist, total_outward_dist)
                total_outward_dist = 0
            moving_outward = False
            dist = curr_dist
    return max_outward_dist

## TODO check these bad larries for bugs
def wrists_inward(gesture):
    r_hand = get_rl_hand_keypoints(gesture, 'r')
    l_hand = get_rl_hand_keypoints(gesture, 'l')
    moving_inward = False
    total_inward_dist = 0
    max_inward_dist = 0
    dist = abs(avg(r_hand[0]['x']) - avg(l_hand[0]['x']))
    for i in range(0, len(r_hand)-1):
        curr_dist = abs(avg(r_hand[i]['x']) - avg(l_hand[i]['x']))
        if curr_dist <= dist:
            moving_inward = True
            total_inward_dist = total_inward_dist + abs(curr_dist - dist)
            dist = curr_dist
        else:
            if moving_inward:
                max_inward_dist = max(max_inward_dist, total_inward_dist)
                total_inward_dist = 0
            moving_inward = False
            dist = curr_dist
    return max_inward_dist





## across all the frames, how much does it go back and forth?
## basically, how much do movements switch direction? but on a big scale.
## average the amount over the hands
def oscillation(gesture):
    return


def get_gesture_features(gesture):
    gesture_features = [
    #   max_vert_accel,
    #   max_horiz_accel,
      palm_vert(gesture, 'l'),
      palm_horiz(gesture, 'l'),
      palm_vert(gesture, 'r'),
      palm_horiz(gesture, 'r'),
      max_hands_apart(gesture),
      min_hands_together(gesture),
    #   x_oscillate,
    #   y_oscillate,
      wrists_up(gesture, 'r'),
      wrists_up(gesture, 'l'),
      wrists_down(gesture, 'r'),
      wrists_down(gesture, 'l'),
    #   wrists_outward,
    #   wrists_inward,
    #   wrists_sweep,
    #   wrist_arc,
    #   r_hand_rotate,
    #   l_hand_rotate,
    #   hands_cycle
    ]
    return gesture_features






########################################################
####################### Helpers ########################
########################################################

def avg(v):
    return float(sum(v) / len(v))

def get_body_keypoints(gesture):
    return get_keypoints_body_range(gesture, 0, 7)

def get_rl_hand_keypoints(gesture, hand):
    if hand == 'r':
        return get_keypoints_body_range(gesture, 7, 28)
    elif hand == 'l':
        return get_keypoints_body_range(gesture, 28, 49)

def get_keypoints_body_range(gesture, start, end):
    keys = []
    for t in gesture['keyframes']:
        y = t['y'][start:end]
        x = t['x'][start:end]
        keys.append({'y': y, 'x': x})
    return keys

def calculate_distance_between_gestures(g1, g2):
    feat1 = np.array(get_gesture_features(g1))
    feat2 = np.array(get_gesture_features(g2))
    return np.linalg.norm(feat1-feat2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-gesture_data', '--gesture_data', help='json of all gesture motion data', required=True)
    parser.add_argument('-seed_gestures', '--seed_gestures', help='list of gesture ids that will serve as starting seed gestures', required=False)

    print "processing data"

    all_gesture_data = args.gesture_data
    seed_gesture_ids = args.seed_gestures
