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
import uuid
import operator
from analyze_frames import *


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

# returns minimum distance at any frame between point A on right hand and
# point A on left hand.
def min_hands_together(gesture):
    return hand_togetherness(gesture, 1000, min)

# returns maximum distance at any frame between point A on right hand and
# point A on left hand
def max_hands_apart(gesture):
    return hand_togetherness(gesture, 0, max)

def hand_togetherness(gesture, min_max, relate):
    r_hand_keys = get_rl_hand_keypoints(gesture, 'r')
    l_hand_keys = get_rl_hand_keypoints(gesture, 'l')
    max_dist = min_max # larger than pixel range
    for i in range(0, len(r_hand_keys)-2):
        for j in range(0, len(r_hand_keys[i]['x'])-1):
            r_x = r_hand_keys[i]['x'][j]
            r_y = r_hand_keys[i]['y'][j]
            l_x = l_hand_keys[i]['x'][j]
            l_y = l_hand_keys[i]['y'][j]
            a = np.array((r_x, r_y))
            b = np.array((l_x, l_y))
            dist = np.linalg.norm(a-b)
            max_dist = relate(dist, max_dist)
    return max_dist

# get maximum "verticalness" aka minimum horizontalness of hands
def palm_vert(gesture, lr):
    return palm_angle_axis(gesture, lr, 'x')

# get maximum "horizontalness" aka minimum verticalness of hands
def palm_horiz(gesture, lr):
    return palm_angle_axis(gesture, lr, 'y')

def palm_angle_axis(gesture, lr, xy):
    hand_keys = get_rl_hand_keypoints(gesture, lr)
    p_min = 1000
    for frame in hand_keys:
        max_frame_dist = max(frame[xy]) - min(frame[xy])
        p_min = min(p_min, max_frame_dist)
    return p_min

## max distance from low --> high the wrists move in a single stroke
def wrists_up(gesture, lr):
    return wrist_vertical_stroke(gesture, lr, operator.ge)

## max distance from high --> low the wrists move in single stroke
def wrists_down(gesture, lr):
    return wrist_vertical_stroke(gesture, lr, operator.le)

def wrist_vertical_stroke(gesture, lr, relate):
    hand_keys = get_rl_hand_keypoints(gesture, lr)
    total_motion = 0
    max_single_stroke = 0
    pos = avg(hand_keys[0]['y'])
    same_direction = False
    for frame in hand_keys:
        curr_pos = avg(frame['y'])
        if relate(curr_pos, pos):
            total_motion = total_motion + abs(curr_pos - pos)
            pos = curr_pos
            same_direction = True
        else:
            if same_direction:
                total_motion = 0
            same_direction = False
            pos = curr_pos
        max_single_stroke = max(max_single_stroke, total_motion)
    return max_single_stroke

def wrists_outward(gesture):
    return wrist_relational_move(gesture, operator.ge)

def wrists_inward(gesture):
    return wrist_relational_move(gesture, operator.le)

def wrist_relational_move(gesture, relate):
    r_hand = get_rl_hand_keypoints(gesture, 'r')
    l_hand = get_rl_hand_keypoints(gesture, 'l')
    moving_desired_direction = False
    total_direction_dist = 0
    max_direction_dist = 0
    dist = abs(avg(r_hand[0]['x']) - avg(l_hand[0]['x']))
    for i in range(1, len(r_hand)-1):
        curr_dist = abs(avg(r_hand[i]['x']) - avg(l_hand[i]['x']))
        if relate(curr_dist, dist):
            moving_desired_direction = True
            total_direction_dist = total_direction_dist + abs(curr_dist - dist)
            dist = curr_dist
        else:
            if moving_desired_direction:
                total_direction_dist = 0
            moving_desired_direction = False
            dist = curr_dist
        max_direction_dist = max(max_direction_dist, total_direction_dist)
    return max_direction_dist

def wrists_moving_apart(gesture):
    r_hand = get_rl_hand_keypoints(gesture, 'r')
    l_hand = get_rl_hand_keypoints(gesture, 'l')
    moving_apart = False
    total_apart = 0
    max_dist_apart = 0
    dist = get_point_dist(avg(r_hand[0]['x']), avg(r_hand[0]['y']), avg(l_hand[0]['x']), avg(l_hand[0]['y']))
    for i in range(1, len(r_hand)-1):
        curr_dist = get_point_dist(avg(r_hand[i]['x']), avg(r_hand[i]['y']), avg(l_hand[i]['x']), avg(l_hand[i]['y']))
        if curr_dist >= dist:
            moving_apart = True
            total_apart = total_apart + abs(curr_dist - dist)
            dist = curr_dist
        else:
            if moving_apart:
                total_apart = 0
            moving_inward = False
            dist = curr_dist
        max_dist_apart = max(max_dist_apart, total_apart)
    return max_dist_apart

# max velocity over n frames.
# only goes over r/l hand avg pos
def max_wrist_velocity(gesture, rl='r', num_frames=5):
    hand_keys = get_rl_hand_keypoints(gesture, rl)
    max_frame_diff = 0
    for i in range(num_frames, len(hand_keys)-1):
        frame_diff = 0
        for j in range(1, num_frames):
            pos1 = get_hand_pos(hand_keys[i-(j-1)])
            pos2 = get_hand_pos(hand_keys[i-j])
            frame_diff = frame_diff + get_point_dist(pos1[0], pos1[1], pos2[0], pos2[1])
        max_frame_diff = max(max_frame_diff, frame_diff)
    return max_frame_diff


def get_hand_pos(hand_keys):
    return(avg(hand_keys['x']), avg(hand_keys['y']))

def get_point_dist(x1,y1,x2,y2):
    a = np.array((x1, y1))
    b = np.array((x2, y2))
    return np.linalg.norm(a-b)


## across all the frames, how much does it go back and forth?
## basically, how much do movements switch direction? but on a big scale.
## average the amount over the hands
# def oscillation(gesture):
#     return


def get_gesture_features(gesture):
    gesture_features = [
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
      wrists_outward(gesture),
      wrists_inward(gesture),
      max_wrist_velocity(gesture, 'r'),
      max_wrist_velocity(gesture, 'l')
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


###############################################################
####################### DIY Clustering ########################
###############################################################
class GestureClusters():
    # all the gesture data for gestures we want to cluster.
    # the ids of any seed gestures we want to use for our clusters.
    def __init__(self, all_gesture_data, seeds=[]):
        # I have no idea what best practices are but I'm almost certain this is
        # a gross, disgusting anti-pattern for iterating IDs.
        self.c_id = 0
        self.agd = all_gesture_data
        self.clusters = {}
        self.seed_ids = seeds
        if(len(seeds)):
            for seed_g in seeds:
                g = get_gesture_by_id(seed_g, all_gesture_data)
                cluster_id = self.c_id
                self.c_id = self.c_id + 1
                c = {'cluster_id': cluster_id, 'seed_id': g['id'], 'gestures': [g]}
                self.clusters[cluster_id] = c

    def cluster_gestures(self, gesture_data, max_cluster_distance=False):
        gd = gesture_data if gesture_data else self.agd
        for g in gd:
            (nearest_cluster_id, nearest_cluster_dist) = self._get_shortest_cluster_dist(g)
            if max_cluster_distance and nearest_cluster_dist > max_cluster_distance:
                new_cluster_id = self.c_id
                self.c_id = self.c_id + 1
                c = {'cluster_id': new_cluster_id, 'seed_id': g['id'], 'gestures': [g]}
                self.clusters[new_cluster_id] = c
            else:
                self.clusters[nearest_cluster_id]['gestures'].append(g)

    def report_clusters(self, verbose=False):
        print("Number of clusters: %s" % len(self.clusters))
        return

    def _get_shortest_cluster_dist(self, g):
        shortest_dist = 10000
        nearest_cluster_id = ''
        for k in self.clusters:
            c = self.clusters[k]
            dist = self._get_cluster_dist(g, c)
            if dist < shortest_dist:
                shortest_dist = dist
                nearest_cluster_id = c['cluster_id']
        return (nearest_cluster_id, shortest_dist)

    def _get_cluster_dist(self, g, c):
        shortest_dist = 10000   # higher than dist could be
        for c_gesture in c['gestures']:
            shortest_dist = min(shortest_dist, calculate_distance_between_gestures(g, c_gesture))
        return shortest_dist


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-gesture_data', '--gesture_data', help='json of all gesture motion data', required=True)
    parser.add_argument('-seed_gestures', '--seed_gestures', help='list of gesture ids that will serve as starting seed gestures', required=False)

    print "processing data"

    all_gesture_data = args.gesture_data
    seed_gesture_ids = args.seed_gestures
