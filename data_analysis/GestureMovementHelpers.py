import operator
from common_helpers import *
import time
import numpy as np
import statistics
from matplotlib import cm, pyplot as plt
from scipy import optimize
from math import sqrt


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


# draw the path of wrist and each fingertip across n frames.
# used to visualize cycles.
def draw_finger_tip_path_across_frames(handed_keys, starting_frame, n=10, do_plot=True):
    # across all frames
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

    if do_plot:
        plt.plot(fingers['base']['x'], fingers['base']['y'], color=fingers['base']['color'])
        plt.plot(fingers[0]['x'], fingers[0]['y'], color=fingers[0]['color'])
        plt.plot(fingers[1]['x'], fingers[1]['y'], color=fingers[1]['color'])
        plt.plot(fingers[2]['x'], fingers[2]['y'], color=fingers[2]['color'])
        plt.plot(fingers[3]['x'], fingers[3]['y'], color=fingers[3]['color'])
        plt.plot(fingers[4]['x'], fingers[4]['y'], color=fingers[4]['color'])

        plt.show()

    return fingers


def detect_cycle(handed_keys, cycle_length=15):
    scores = []
    min_score = 100000
    min_frame = 0
    for i in range(len(handed_keys)-cycle_length):
        score = evaluate_cycle(handed_keys, i, cycle_length=cycle_length, do_plot=False)
        scores.append((i, score))
        if score < min_score:
            min_frame = i
            min_score = score
    print("best cycle start: ", min_frame)
    print("best cycle score: ", min_score)

    # and then see that the best scoring frames are near each other
    scores.sort(key=lambda x: x[1])
    top_scores = compact_and_sort_cycle_scores(scores)
    return top_scores[:3]


# check how many of the numbers within n of x are in top 25%
def compact_and_sort_cycle_scores(scores, n=8):
    cutoff = int(len(scores) / 4)
    top_tier = scores[:cutoff]
    top_scores = []
    exclude = []
    for e in top_tier:
        if e[0] in exclude:
            continue
        incl = 0
        for i in range(-n, n):
            if e[0] + i in [e[0] for e in top_tier]:
                exclude.append(e[0]+i)
                incl += 1
        top_scores.append((e[0], incl / (n * 2), e[1]))
        top_scores.sort(key=lambda x: x[1], reverse=True)
    return top_scores


def detect_worst_cycle(handed_keys, cycle_length=15):
    max_score = 0
    max_frame = 0
    for i in range(len(handed_keys)):
        score = evaluate_cycle(handed_keys, i, cycle_length=cycle_length, do_plot=False)
        if score > max_score:
            max_frame = i
            max_score = score
    print("best cycle start: ", max_frame)
    print("best cycle score: ", max_score)


def evaluate_cycle(handed_keys, starting_frame, cycle_length=15, do_plot=True):
    # see that start and end point are within some margin of error
    # that probably depends on the width of the cycle -- say 20% of largest distance?
    fingers = draw_finger_tip_path_across_frames(handed_keys, starting_frame, cycle_length, do_plot=do_plot)
    scores = get_finger_sqrs(fingers, do_plot=do_plot)
    return scores.mean()
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


def draw_middle_circle(xs, ys, color='gray', alpha=0.3):
    r = get_average_distance(xs, ys) / 1.21 # this is dumb circle math.
    px = max(xs) - (max(xs) - min(xs))/2
    py = max(ys) - (max(ys) - min(ys))/2
    c = plt.Circle((px, py), r, color=color, alpha=alpha)
    plt.gcf().gca().add_artist(c)


# quick helper 15 4 20
# will not work to use lstsqr as measure, bc doesn't measure dist to circle EDGE which I think is what we need.
# but this favors smaller motions, need to adjust for how many pixels the whole thing takes up.
def get_finger_sqrs(fingers, do_plot=True):
    if do_plot:
        for i in range(0, 6):
            if i == 5:
                draw_middle_circle(fingers['base']['x'], fingers['base']['y'], color=fingers['base']['color'])
                continue
            draw_middle_circle(fingers[i]['x'], fingers[i]['y'], color=fingers[i]['color'])

    lbase = calculate_cycle_score(fingers['base']['x'], fingers['base']['y'])
    l0 = calculate_cycle_score(fingers[0]['x'], fingers[0]['y'])
    l1 = calculate_cycle_score(fingers[1]['x'], fingers[1]['y'])
    l2 = calculate_cycle_score(fingers[2]['x'], fingers[2]['y'])
    l3 = calculate_cycle_score(fingers[3]['x'], fingers[3]['y'])
    l4 = calculate_cycle_score(fingers[4]['x'], fingers[4]['y'])

    lbase_angles = calculate_angle_score(fingers['base']['x'], fingers['base']['y'])
    l0_angles = calculate_angle_score(fingers[0]['x'], fingers[0]['y'])
    l1_angles = calculate_angle_score(fingers[1]['x'], fingers[1]['y'])
    l2_angles = calculate_angle_score(fingers[2]['x'], fingers[2]['y'])
    l3_angles = calculate_angle_score(fingers[3]['x'], fingers[3]['y'])
    l4_angles = calculate_angle_score(fingers[4]['x'], fingers[4]['y'])

    return np.array([lbase_angles, l0_angles, l1_angles, l2_angles, l3_angles, l4_angles])
    # return np.array([lbase, l0, l1, l2, l3, l4])


# need to get least squares circle.
# TODO make this get distance to EDGE of circle!!
def calculate_cycle_score(xs, ys):
    # make a circle that is squiggly but pretty good for true cycles.
    # TODO maybe this has to be more ovular.

    # penalize paths that don't go in a circle.      # TODO penalize angles < 90? -- yes, specifically the more
    # we're going polar.                             # TODO bad it is, the more we need to penalize it (lower from 90)
    xs = np.array(xs)                                # TODO maybe just penalize that there are DIFFERENT angles?
    ys = np.array(ys)
    # fuck it can't convert polar coordinates to new origin, I'll just convert the cartesian coords FIRST.
    path_direction_score = calculate_path_direction_score(xs, ys)
    angle_score = calculate_angle_score(xs, ys)
    return path_direction_score


def calculate_angle_score(xs, ys):
    angles = []
    for i in range(1, len(xs)-1):
        a = np.array([xs[i-1], ys[i-1]])
        b = np.array([xs[i], ys[i]])
        c = np.array([xs[i+1], ys[i+1]])
        angle = _calculate_angle(a, b, c)
        if math.isnan(angle):
            angle = 0
        angles.append(angle)
    print(angles)
    angles = np.array(angles)
    print(angles.std())
    print(angles.mean())
    # multiply by velocity because if the hand isn't moving these will both be 0
    return angles.std()


def calculate_path_direction_score(xs, ys):
    px = max(xs) - (max(xs) - min(xs))/2
    py = max(ys) - (max(ys) - min(ys))/2
    Ri = calc_r(px, py, np.array(xs), np.array(ys))
    least_sqr = Ri.mean()
    pol_xs = xs - px
    pol_ys = ys - py
    pols = [cart2pol(pol_xs[i], pol_ys[i]) for i in range(len(xs))]
    rs = np.array([p[0] for p in pols])
    thetas = [p[1] for p in pols]       # check these are continuous, penalize if not.
    dirs = same_dir_theta(thetas)    # just want them going in same direction, don't really care which way.
    bonus = 0
    for i in range(1, len(dirs)):
        if (dirs[i] == dirs[i-1]) and dirs[i] != '-':
            bonus += _get_point_dist(xs[i], ys[i], xs[i-1], ys[i-1])
        else:
            bonus -= _get_point_dist(xs[i], ys[i], xs[i-1], ys[i-1])
    # want a nice circle, in the same direction, with low stdev to center.
    polar_score = least_sqr.mean() - bonus + rs.std()
    return polar_score


def cart2pol(x, y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return(rho, phi)


def pol2cart(rho, phi):
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return(x, y)


# WE'E ONTO SOMETHING HERE.
# can tell if something IS a cycle
# can also be used to detect cycles (when you get lots of the same direction in a row)
# problem with that is centering the coordinates around wherever the center circle is
def same_dir_theta(ts):
    same_dir = []
    for i in range(1, len(ts)-1):
        if ts[i-1] < ts[i] < ts[i+1]:
            same_dir.append('d')
        elif ts[i-1] > ts[i] > ts[i+1]:
            same_dir.append('u')
        else:
            same_dir.append('-')
    return same_dir


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
    deg = np.degrees(ang_b)
    # if math.isnan(deg):
    #     deg = 0             # this happens when 2 or more of the 3 points are the same.
    return deg


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


# get least squares circle
# https://scipy-cookbook.readthedocs.io/items/Least_Squares_Circle.html
def calc_r(xc, yc, xs, ys):
    """ calculate the distance of each 2D points from the center (xc, yc) """
    return np.sqrt((xs-xc)**2 + (ys-yc)**2)


def f_2(cx, cy, xs, ys):
    """ calculate the algebraic distance between the data points and the mean circle centered at c=(xc, yc) """
    Ri = calc_r(cx, cy, xs, ys)
    return Ri - Ri.mean()


def get_leastsqr_circle(xs, ys):
    # coordinates of the barycenter
    x_m = statistics.mean(xs)
    y_m = statistics.mean(ys)
    center_estimate = x_m, y_m
    center_2, ier = optimize.leastsq(lambda c: f_2(c[0], c[1], xs, ys),
                                     center_estimate)

    xc_2, yc_2 = center_2
    Ri_2 = calc_r(xc_2, yc_2, xs, ys)
    R_2 = Ri_2.mean()
    least_sqr = sum((Ri_2 - R_2) ** 2)
    return xc_2, yc_2, least_sqr
