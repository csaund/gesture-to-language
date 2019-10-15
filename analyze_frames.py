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


########################################################
############### Getting Actual Keyframes ###############
########################################################

# takes time in form of "X_X_IND_m_s.txt"
def timestring_to_int(time):
    times = time.split("_")
    hrs = int(times[-3])
    mins = int(times[-2])
    sec_and_txt_arr = times[-1].split(".")
    secs = float(sec_and_txt_arr[0])
    if(len(sec_and_txt_arr) == 3):
        secs = secs + float(str("." + str(sec_and_txt_arr[1])))
    timesec = (hrs * 60 * 60) + (mins * 60) + secs
    return timesec

# takes start time in form of X_X_IND_m_s.txt, end time in form of X_X_IND_m_s.txt
# and question time in form of X_X_IND_m_s.txt
# returns whether or not question time is between start and end times
def is_within_time(start_time, end_time, question_time):
    s = timestring_to_int(start_time)
    e = timestring_to_int(end_time)
    q = timestring_to_int(question_time)
    if (q < s) or (q > e):
        return False
    return True

# takes fp to text file, returns dataframe of
# csv version of that text file
# however it takes the x y and zips them together to make
# a big vector of x,y pairs.
def extract_txt_data(bp, filepath):
    fn = filepath.split('.txt')[0]
    inf = fn + '.txt'
    outf = fn + '.csv'
    with open(bp + inf, 'r') as in_file:
        lines = in_file.read().splitlines()
        stripped = [line.replace(","," ").split() for line in lines]
        zipped = zip(stripped[1], stripped[2])
        with open(bp + outf, 'w') as out_file:
            writer = csv.writer(out_file)
            writer.writerow(('x', 'y'))
            i = 0
            for z in zipped:
                row = (str(z[0]), str(z[1]))
                writer.writerow(row)
                i += 1
    # get the data that we need
    dat = pd.read_csv(str(bp + outf))
    # clean up after ourselves
    os.remove(str(bp + outf))
    return dat          # output is a written csv to that location

# takes number of seconds (408.908909) and converts to something like
# MM_S.SS
# that is searchable in the folder.
def get_filekey(t):
    keyframe_min = int(math.floor(t / 60))
    keyframe_sec = round(t - (keyframe_min * 60), 6)
    ## add leading 0s to avoid clashes.
    if keyframe_min < 10:
        keyframe_min = '0' + str(keyframe_min)
    if keyframe_sec < 10:
        keyframe_sec = '0' + str(keyframe_sec)
    filekey = str(keyframe_min) + '_' + str(keyframe_sec)
    return filekey


## requires mapping from video path to specific path.
## ex this needs /Users/carolynsaund/github/gest-data/data/rock/keypoints_simple/1
## somehow successfully returns the keyframe data for that gesture.
# gesture vid path is something like "gest-data/data/rock/keypoints_simple/1/"
def get_keyframes_per_gesture(gesture_video_path, start_time, end_time):
    start_filekey = str(get_filekey(start_time))
    end_filekey = str(get_filekey(end_time))
    s_key = "X_X_0_" + start_filekey + ".txt"  # I am an absolute LUG. This is for simplicity in file format to get times for each file
    e_key = "X_X_0_" + end_filekey + ".txt"
    print("s_key: %s" % s_key)
    print("e_key: %s" % e_key)
    files = sorted(os.listdir(gesture_video_path), key=timestring_to_int)
    m = [s for s in files if start_filekey in s]
    if len(m) != 1:
        print("panic!! wrong number of matching files: %s" % str(len(m)))
        print(start_filekey)
        print(m)
        return
    all_gesture_keys = []
    i = files.index(m[0])
    print("starting at %s" % files[i])   # start at index of first frame
    while is_within_time(s_key, e_key, files[i]):
        dat = extract_txt_data(gesture_video_path, files[i])
        all_gesture_keys.append({'x': list(dat['x']), 'y': list(dat['y'])}) # TODO change this to pd
        i+=1
        if(i >= len(files)):
            print("WARNING: GOING BEYOND KEYPOINT TIMES: %s" % str(files[i-1]))
            break
    print("ending at %s" % files[i-1]) # the -1 is a hack until I figure out why there's missing keypoint data
    return all_gesture_keys



########################################################
################ All the plotting stuff ################
########################################################

## flip so instead of format like
# [t1, t2, t3], [t1`, t2`, t3`], [t1``, t2``, t3``]
# it's in the format of
# [t1, t1`, t1``], [t2, t2`, t2``], [t3, t3`, t3``]
def arrange_data_by_time(dat_vector):
    flipped_dat = []
    for i in range(len(dat_vector[0])):
        a = []
        for d in dat_vector:
            a.append(d[i])
        flipped_dat.append(a)
    return flipped_dat


def plot_both_gesture_coords(gesture):
    plt.subplot(1,2,1)
    plot_coords('x', gesture)
    plt.subplot(1,2,2)
    plot_coords('y', gesture)
    plt.title = 'xy coordinates for gesture %s' % gesture['id']
    plt.savefig('%s.png' % gesture['id'])
    plt.show()
    return


def plot_coords(x_y, gesture):
    coords = coords = [d[x_y] for d in gesture['keyframes']]
    fc = arrange_data_by_time(coords)
    for v in fc:
        plt.plot(range(0, len(fc[0])), v)
    plt.xlabel("frame")
    plt.ylabel("%s pixel position" % x_y)
    plt.title = '%s coordinates for gesture %s' % (x_y, gesture['id'])


## let's follow 88279 through and see where the bugs are...
def analyze_gestures(video_base_path, timings_path):
    all_gesture_data = []
    timings = get_timings(timings_path)
    l = len(timings['phrases'])
    i = 0
    for phase in timings['phrases']:
        i = i + 1
        print("%s / %s gesutres processed." % (i, l))
        p = phase['phase']
        vid_path = video_base_path + str(p['video_fn'].split('.')[0]) + '/'
        start = p['start_seconds']
        end = p['end_seconds']
        specific_gesture_dat = {'id': phase['id']}
        print("processing gesture id %s" % phase['id'])
        specific_gesture_dat['keyframes'] = get_keyframes_per_gesture(vid_path, start, end)
        all_gesture_data.append(specific_gesture_dat)
    return all_gesture_data

# takes [id1, id2, id3] and saves
# the plot images of those ids
def save_gesture_plots(gesture_ids, all_gestures):
    for i in gesture_ids:
        g = get_gesture_by_id(i, all_gestures)
        plot_both_gesture_coords(i)
    return

def plot_dist_of_num_frames_by_gesture(all_gestures):
    num_frames = []
    for g in all_gestures:
        num_frames.append(len(g['keyframes']))
    num_bins = 5
    n, bins, patches = plt.hist(num_frames, bins=[0,50,100,200,400,2500], facecolor='blue', alpha=0.5)
    plt.title = "number of frames per gesture"
    plt.show()

########################################################
####################### Helpers ########################
########################################################
# ex. id=73848, return {id: 73848, keyframes: [{x: [...], y: [...], ...]}}
def get_gesture_by_id(d_id, all_speaker_gesture_data):
    dat = [d for d in all_speaker_gesture_data if d['id'] == d_id]
    # because this returns list of matching items, and only one item will match,
    # we just take the first element and use that.
    return dat[0]

def get_timings(timings_path):
    with open(timings_path) as f:
        timings = json.load(f)
    return timings

def is_within_time_ez(s, e, q):
    if q < s or q > e:
        return False
    return True

#88279
def get_single_gesture_from_timings(gesture_id, video_base_path, timings_path):
    gesture_data = {}
    timings = get_timings(timings_path)
    for phase in timings['phrases']:
        p = phase['phase']
        if phase['id'] == gesture_id:
            start = p['start_seconds']
            end = p['end_seconds']
            vid_path = video_base_path + str(p['video_fn'].split('.')[0]) + '/'
            gesture_data['id'] = phase['id']
            gesture_data['keyframes'] = get_keyframes_per_gesture(vid_path, start, end)
    return gesture_data

# takes time in MM_SS.MS
def get_gesture_ids_from_video_and_timing(video_fn_id, time_in_mm_ss, timings_path):
    t = get_timings(timings_path)
    candidates = []
    # transform a reasonable time to a dumb time.
    t_to_trans = "X_X_0_%s.txt" % time_in_mm_ss
    time_in_seconds = timestring_to_int(t_to_trans)
    all_video_gestures = [p for p in tim['phrases'] if p['phase']['video_fn'] == video_fn_id]
    for gesture in all_video_gestures:
        p = gesture['phase']
        if is_within_time_ez(p['start_seconds'], p['end_seconds'], time_in_seconds):
            candidates.append(gesture['id'])
    return candidates

# get number of frames for a particular gesture by gesture id
def get_num_frames(gesture_id, all_gesture_data):
    g = get_gesture_by_id(gesture_id, all_gesture_data)
    return len(g['keyframes'])

## helper, so can use all these random vars I have floating around I guess
def get_plot_by_video_time(video_fn_id, time_in_mm_ss, video_base_path, timings_path):
    ids = get_gesture_ids_from_video_and_timing(video_fn_id, time_in_mm_ss, timings_path)
    if len(ids) > 1:
        print "more than one gesture associated with this time, using gesture %s" % ids[0]
    i = ids[0]
    print i
    gest = get_single_gesture_from_timings(i, video_base_path, timings_path)
    plot_both_gesture_coords(gest)


################################################################################
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-base_path', '--base_path', help='base folder path of dataset', required=True)
    #parser.add_argument('-output_path', '--output_path', default='output directory to save wav files', required=True)
    parser.add_argument('-speaker', '--speaker', default='optionally, run only on specific speaker', required=False)

    args = parser.parse_args()
    video_path = args.base_path + '/' + args.speaker + '/keypoints_simple/'
    timings_path = args.base_path + '/' + args.speaker + '/timings.json'

    print "processing data"
    all_speaker_gesture_keypoints = analyze_gestures(video_path, timings_path)



## TODO write unit tests for those bad boys up above
