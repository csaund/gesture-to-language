import operator
from common_helpers import *
import time
import numpy as np
import statistics
from matplotlib import cm, pyplot as plt

BASE_KEYPOINT = [0]
# DO NOT TRUST THESE
RIGHT_BODY_KEYPOINTS = [1, 2, 3, 31]
LEFT_BODY_KEYPOINTS = [4, 5, 6, 10]
RIGHT_WRIST_KEYPOINT = 3
LEFT_WRIST_KEYPOINT = 6
LEFT_HAND_KEYPOINTS = lambda x: [10] + [11 + (x * 4) + j for j in range(4)]  # CHECK THESE
RIGHT_HAND_KEYPOINTS = lambda x: [31] + [32 + (x * 4) + j for j in range(4)]   # CHECK THESE
ALL_RIGHT_HAND_KEYPOINTS = list(range(31, 52))
ALL_LEFT_HAND_KEYPOINTS = list(range(10, 31))
BODY_KEYPOINTS = RIGHT_BODY_KEYPOINTS + LEFT_BODY_KEYPOINTS
DIRECTION_ANGLE_SWITCH = 110  # arbitrary measure of degrees to constitute hands switching direction ¯\_(ツ)_/¯

cluster_method_timings = {'test': 0.0}


# https://medium.com/pythonhive/python-decorator-to-measure-the-execution-time-of-methods-fa04cb6bb36d
def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        else:
            n = method.__name__
            if n in cluster_method_timings.keys():
                cluster_method_timings[n] = cluster_method_timings[n] + (te-ts) * 1000
            else:
                cluster_method_timings[n] = (te-ts) * 1000
        return result
    return timed


##########################################################################
# STATIC MOTION FUNCTIONS ################################################
##########################################################################
# get maximum "verticalness" aka minimum horizontalness of hands
def _palm_vert(keyframes):
    return _palm_angle_axis(keyframes, 'x')


# get maximum "horizontalness" aka minimum verticalness of hands
def _palm_horiz(keyframes):
    return _palm_angle_axis(keyframes, 'y')


@timeit
def _palm_angle_axis(keyframes, xy):
    p_min = 1000
    for frame in keyframes:
        max_frame_dist = max(frame[xy]) - min(frame[xy])
        p_min = min(p_min, max_frame_dist)
    return p_min


# max distance from low --> high the wrists move in a single stroke
def _wrists_up(keyframes):
    return _wrist_vertical_stroke(keyframes, operator.ge)


# max distance from high --> low the wrists move in single stroke
def _wrists_down(keyframes):
    return _wrist_vertical_stroke(keyframes, operator.le)


@timeit
def _wrist_vertical_stroke(keyframes, relate):
    total_motion = 0
    max_single_stroke = 0
    # 0 is the index of the wrist in handed keypoints
    pos = keyframes[0]['y'][0]
    same_direction = False
    for frame in keyframes:
        curr_pos = frame['y'][0]
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


# returns minimum distance at any frame between point A on right hand and
# point A on left hand.
def _min_hands_together(r_hand_keys, l_hand_keys):
    return _max_hand_togetherness(r_hand_keys, l_hand_keys, 10000, min)


# returns maximum distance at any frame between point A on right hand and
# point A on left hand
def _max_hands_apart(r_hand_keys, l_hand_keys):
    return _max_hand_togetherness(r_hand_keys, l_hand_keys, 0, max)


@timeit
def _max_hand_togetherness(r_hand_keys, l_hand_keys, min_max, relate):
    max_dist = min_max          # larger than pixel range
    for frame_index in range(len(r_hand_keys)):     # for each frame in the gesture, same cause it's the same gesture!
        # determine how far apart r and l hand are, vertically and horizontally
        cur_r_keys = r_hand_keys[frame_index]
        cur_l_keys = l_hand_keys[frame_index]
        r_pos = np.array((cur_r_keys['x'], cur_r_keys['y']))
        l_pos = np.array((cur_l_keys['x'], cur_l_keys['y']))
        dist = np.linalg.norm(r_pos-l_pos)
        max_dist = relate(dist, max_dist)
    return max_dist


# measures the largest outward motion of r/l wrists
# that is, the largest distance in which wrists are moving
# continuously apart.
# THIS IS IN SPACE, USES BOTH HORIZ AND VERT AXES
def _wrists_apart(r_hand_keys, l_hand_keys):
    return _wrist_togetherness(r_hand_keys, l_hand_keys, operator.ge)


def _wrists_together(r_hand_keys, l_hand_keys):
    return _wrist_togetherness(r_hand_keys, l_hand_keys, operator.le)


@timeit
def _wrist_togetherness(r_hand_keys, l_hand_keys, relate):
    total_direction_dist = 0
    max_direction_dist = 0
    # the 0th keyframe of each hand is the wrist position
    r_wrist_position = np.array([r_hand_keys[0]['x'][0], r_hand_keys[0]['y'][0]])
    l_wrist_position = np.array([l_hand_keys[0]['x'][0], l_hand_keys[0]['y'][0]])
    prev_dist = np.linalg.norm(r_wrist_position - l_wrist_position)
    for frame_index in range(len(r_hand_keys)):
        r_wrist_position = np.array([r_hand_keys[frame_index]['x'][0], r_hand_keys[frame_index]['y'][0]])
        l_wrist_position = np.array([l_hand_keys[frame_index]['x'][0], l_hand_keys[frame_index]['y'][0]])
        cur_dist = np.linalg.norm(r_wrist_position - l_wrist_position)
        if relate(cur_dist, prev_dist):     # we are moving in the desired direction
            total_direction_dist += abs(cur_dist - prev_dist)
            prev_dist = cur_dist
            max_direction_dist = max(max_direction_dist, total_direction_dist)
        else:                       # we're not moving in the desired direction
            total_direction_dist = 0
            prev_dist = cur_dist
    return max_direction_dist


# max velocity of wrist between 2 frames
# specifically, it just gets the max difference in distance between wrists across 2 frames
@timeit
def _max_wrist_velocity(keys):
    max_dist = 0
    for i in range(len(keys)-1):
        # wrist is 0th keypoint for each hand
        (wx0, wy0) = (keys[i]['x'][0], keys[i]['y'][0])
        (wx1, wy1) = (keys[i+1]['x'][0], keys[i+1]['y'][0])
        max_dist = max(max_dist, _get_point_dist(wx0, wy0, wx1, wy1))
    return max_dist


# measures the number of times wrist changes direction as measured by the angle
# of the wrist point between frame a, b, c is greater than 100
def _get_back_and_forth(keys):
    switches = 0
    if len(keys) < 3:
        return 0
    for frame in range(len(keys)-2):
        a = np.array([keys[frame]['x'][0], keys[frame]['y'][0]])
        b = np.array([keys[frame+1]['x'][0], keys[frame+1]['y'][0]])
        c = np.array([keys[frame+2]['x'][0], keys[frame+2]['y'][0]])
        ba = a - b
        bc = c - b
        cos_ang = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        full_ang = np.arccos(cos_ang)
        if np.degrees(full_ang) >= DIRECTION_ANGLE_SWITCH:      # arbitrary measure for switching directions ¯\_(ツ)_/¯
            switches += 1
    return switches


def _detect_cycles(keys):
    return keys
    # what happens in a cycle?
    # the fingers rotate around the wrist
    # the angle of each of the fingers stays relatively constant
    # the wrist stays in relatively the same place, or moves slowly
    # the fingers all have relatively the same angle?
    # tips all make a rough circle shape
    # plot finger TIP path between frames, observe
    # gest 1726
    # check if fingertip points are within certain distance of a circle?


# draw the path of wrist and each fingertip across n frames.
# used to visualize cycles.
def draw_finger_tip_path_across_frames(handed_keys, starting_frame, n=10):
    # across all frames
    #for frame_index in range(0, len(handed_keys)-n):
    if len(handed_keys) < (starting_frame + n):
        print(starting_frame + n)
        print("Attempting to go outside of keyframe range. Will only go to frame %s" % len(handed_keys))
        n = len(handed_keys) - starting_frame
    fingers = {
        'base': {
            'x': [],
            'y': [],
            'color': 'gray'
        },
        0: {
            'x': [],
            'y': [],
            'color': 'red'
        },
        1: {
            'x': [],
            'y': [],
            'color': 'blue'
        },
        2: {
            'x': [],
            'y': [],
            'color': 'green'
        },
        3: {
            'x': [],
            'y': [],
            'color': 'orange'
        },
        4: {
            'x': [],
            'y': [],
            'color': 'purple'
        },
    }

    for k in range(starting_frame, starting_frame + n):
        kf = handed_keys[k]
        for i in range(5):
            # base_pos = np.array((kf['x'][0], kf['y'][0]))
            # tip_pos = np.array((kf['x'][(i * 4) + 4], kf['y'][(i * 4) + 4]))
            fingers['base']['x'].append(kf['x'][0])
            fingers['base']['y'].append(kf['y'][0])
            fingers[i]['x'].append(kf['x'][(i * 4) + 4])
            fingers[i]['y'].append(kf['y'][(i * 4) + 4])

    plt.plot(fingers['base']['x'], fingers['base']['y'], color=fingers['base']['color'])
    plt.plot(fingers[0]['x'], fingers[0]['y'], color=fingers[0]['color'])
    plt.plot(fingers[1]['x'], fingers[1]['y'], color=fingers[1]['color'])
    plt.plot(fingers[2]['x'], fingers[2]['y'], color=fingers[2]['color'])
    plt.plot(fingers[3]['x'], fingers[3]['y'], color=fingers[3]['color'])
    plt.plot(fingers[4]['x'], fingers[4]['y'], color=fingers[4]['color'])

    plt.show()
    return fingers


def detect_cycles(handed_keys, starting_frame, est_cycle_length=15):
    # see that start and end point are within some margin of error
    # that probably depends on the width of the cycle -- say 20% of largest distance?
    fingers = draw_finger_tip_path_across_frames(handed_keys, starting_frame, est_cycle_length)
    finger = fingers[3]
    cx, cy, r = draw_middle_path_circle(finger['x'], finger['y'], finger['color'])
    ps = [(finger['x'][i], finger['y'][i]) for i in range(len(finger['x']))]
    dists = [dist_btw_point_and_circle(cx, cy, r, ps[i][0], ps[i][1]) for i in range(len(ps))]
    return statistics.mean(dists)
    # draw a circle defined by starting and furthest point
    # calculate distance of all points from that circle (for JUST that finger)


# for all points (x,y) in xs, ys, return maximum distance between any two.
def get_furthest_distance(xs, ys):
    if len(xs) != len(ys):
        print("unequal length of xs and ys. Unable to calculate max distance")
        return None
    ps = [(xs[i], ys[i]) for i in range(len(xs))]
    max_dist = 0
    for i in range(len(ps)):
        for j in range(i, len(ps)):
            max_dist = max(max_dist, _get_point_dist(ps[i][0], ps[i][1], ps[j][0], ps[j][1]))
    return max_dist


def get_average_distance(xs, ys):
    if len(xs) != len(ys):
        print("unequal length of xs and ys. Unable to calculate max distance")
        return None
    ps = [(xs[i], ys[i]) for i in range(len(xs))]
    dists = []
    for i in range(len(ps)):
        for j in range(i, len(ps)):
            dists.append(_get_point_dist(ps[i][0], ps[i][1], ps[j][0], ps[j][1]))
    return statistics.mean(dists)


def draw_middle_path_circle(xs, ys, color='gray', alpha=0.3):
    # instead of furthest, r should be avg dist between points / 2
    r = get_average_distance(xs, ys) / 1.21 # this is dumb circle math.
    i = int(len(xs) / 2)
    # print(i)
    # print(xs[0], ys[0])
    # print(xs[i], ys[i])
    px = max(xs) - (max(xs) - min(xs))/2
    py = max(ys) - (max(ys) - min(ys))/2
    # px = statistics.mean([xs[0], xs[i]])
    # py = statistics.mean([ys[0], ys[i]])
    # print(px, py)
    # print(r)
    c = plt.Circle((px, py), r, color=color, alpha=alpha)
    plt.gcf().gca().add_artist(c)
    return px, py, r


# https://www.geeksforgeeks.org/shortest-distance-between-a-point-and-a-circle/
def dist_btw_point_and_circle(cx, cy, r, px, py):
    return abs(((((px - cx) ** 2) + ((py - cy) ** 2)) ** (1 / 2)) - r)


# TODO check this in MotionAnalyzerTests and visually.
# actually right now seems to detect smooth gestures???
# definitely not right.
def _max_acceleration(keys):
    max_accel = 0
    for frame in range(len(keys)-2):
        ax, ay = keys[frame]['x'][0], keys[frame]['y'][0]
        bx, by = keys[frame+1]['x'][0], keys[frame+1]['y'][0]
        cx, cy = keys[frame+2]['x'][0], keys[frame+2]['y'][0]
        d1 = _get_point_dist(ax, ay, bx, by)   # /1 for 1 frame (velocity, not distance technically)
        d2 = _get_point_dist(bx, by, cx, cy)
        max_accel = max(max_accel, abs(d1-d2))
    return max_accel


def _plot_hand_angles_across_frame(handed_keys, angle_i=None, smoothing=0):
    xs = []
    ys = []
    for i in range(len(handed_keys)):
        angles = _get_hand_angles_for_frame(handed_keys, i)
        ys.append(angles[angle_i])
        xs.append(i)

    # test trying to smooth....
    if smoothing:
        for i in range(1, len(ys)):
            if abs(ys[i] - ys[i-1]) > smoothing:
                ys[i] = ys[i-1]

    plt.plot(xs, ys)


def _calculate_angle(a, b, c):
    ab = a - b
    cb = c - b
    cos_b = np.dot(ab, cb) / (np.linalg.norm(ab) * np.linalg.norm(cb))
    ang_b = np.arccos(cos_b)
    return np.degrees(ang_b)


def plot_finger_average_across_frames(handed_keys, finger_i=0):
    xs = []
    ys = []
    for i in range(len(handed_keys)):
        angles = _get_average_finger_angles(handed_keys, i)
        ys.append(angles[finger_i])
        xs.append(i)
    # plt.show()
    print(xs)
    print(ys)
    plt.plot(xs, ys)


def _get_average_finger_angles(handed_keys, frame_index):
    angles = _get_hand_angles_for_frame(handed_keys, frame_index)
    fing1 = angles[0:3]
    fing2 = angles[3:6]
    fing3 = angles[6:9]
    fing4 = angles[9:12]
    fing5 = angles[12:15]
    return [statistics.mean(fing1), statistics.mean(fing2), statistics.mean(fing3), statistics.mean(fing4),
            statistics.mean(fing5)]


# given a set of keys from a hand (array length 22), returns angles between every 3 points, like trigrams
# works on frame i
# if hand angles are roughly the same (within like, 20 degrees for each thing) then they're about the same shape
# calculate angles for
# 0,1,2,3,4
# 0,5,6,7,8
# 0,9,10,11,12
# 0,13,14,15,16
# 0,17,18,19,20
# for each of these calc between 0-1-2, 1-2-3, 2-3-4
# TODO map angles against frame...
def _get_hand_angles_for_frame(handed_keys, frame_index):
    angles = []
    kf = handed_keys[frame_index]
    for i in range(5):      # 5 fingers
        base = np.array((kf['x'][0], kf['y'][0]))
        a = np.array((kf['x'][(i*4)+1], kf['y'][(i*4)+1]))
        b = np.array((kf['x'][(i*4)+2], kf['y'][(i*4)+2]))
        c = np.array((kf['x'][(i*4)+3], kf['y'][(i*4)+3]))
        d = np.array((kf['x'][(i*4)+4], kf['y'][(i*4)+4]))
        angles.append(_calculate_angle(base, a, b))
        angles.append(_calculate_angle(a, b, c))
        angles.append(_calculate_angle(b, c, d))
    return angles


@timeit
def _get_point_dist(x1, y1, x2, y2):
    a = np.array((x1, y1))
    b = np.array((x2, y2))
    return np.linalg.norm(a-b)


GESTURE_FEATURES = {
    'palm_vertical': {
        'separate_hands': True,
        'function': _palm_vert
    },
    'palm_horizontal': {
        'separate_hands': True,
        'function': _palm_horiz
    },
    'max_hands_apart': {
        'separate_hands': False,
        'function': _max_hands_apart
    },
    'min_hands_together': {
        'separate_hands': False,
        'function': _min_hands_together
    },
    'wrists_up': {
        'separate_hands': True,
        'function': _wrists_up
    },
    'wrists_down': {
        'separate_hands': True,
        'function': _wrists_down
    },
    'wrists_apart': {
        'separate_hands': False,
        'function': _wrists_apart
    },
    'wrists_together': {
        'separate_hands': False,
        'function': _wrists_together
    },
    'max_wrist_velocity': {
        'separate_hands': True,
        'function': _max_wrist_velocity
    },
    'cycles': {
        'separate_hands': True,
        'function': _detect_cycles
    }
    # 'acceleration': {
    #     'separate_hands': True,
    #     'function': _max_wrist_velocity
    # }
    # 'oscillation': {
    #     'separate_hands': True,
    #     'function': _max_wrist_velocity
    # }
    # 'hand_position_changes': {
    #     'separate_hands': True,
    #     'function': _max_wrist_velocity
    # }
}
