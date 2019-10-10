#!/usr/bin/env pythons
print "importing libs"
import argparse
import pandas as pd
import itertools
import csv
import math
import json
import glob, os

## TODO: make this basepath variable
BASE_PATH = "/Users/carolynsaund/github/gest-data/data/rock/keypoints_simple/1/"


# takes time in form of "X_X_IND_m_s.txt"
def timestring_to_int(time):
    times = time.split("_")
    hrs = int(times[-3])
    mins = int(times[-2])
    sec_and_txt_arr = times[-1].split(".")
    secs = float(sec_and_txt_arr[0]) + float(str("." + str(sec_and_txt_arr[1])))
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
def extract_txt_data(filepath):
    fn = filepath.split('.txt')[0]
    inf = fn + '.txt'
    outf = fn + '.csv'
    with open(BASE_PATH + inf, 'r') as in_file:
        lines = in_file.read().splitlines()
        stripped = [line.replace(","," ").split() for line in lines]
        zipped = zip(stripped[1], stripped[2])
        with open(BASE_PATH + outf, 'w') as out_file:
            writer = csv.writer(out_file)
            writer.writerow(('x', 'y'))
            i = 0
            for z in zipped:
                row = (str(z[0]), str(z[1]))
                writer.writerow(row)
                i += 1
    # get the data that we need
    dat = pd.read_csv(str(BASE_PATH + outf))
    # clean up after ourselves
    os.remove(str(BASE_PATH + outf))
    return dat          # output is a written csv to that location

# takes number of seconds (408.908909) and converts to something like
# MM_S.SS
# that is searchable in the folder.
# TODO make this add the leading 0 before single-digit seconds
# ex 1_4.464464 should be 1_04.464464
def get_filekey(t):
    keyframe_min = int(math.floor(t / 60))
    keyframe_sec = round(t - (keyframe_min * 60), 6)
    filekey = str(keyframe_min) + '_' + str(keyframe_sec)
    return filekey


## requires mapping from video path to specific path.
## ex this needs /Users/carolynsaund/github/gest-data/data/rock/keypoints_simple/1
## somehow successfully returns the keyframe data for that gesture.
def get_keyframes_per_gesture(gesture_video_path, start_time, end_time):
    start_filekey = str(get_filekey(start_time))
    end_filekey = str(get_filekey(end_time))
    # I am an absolute LUG. This is for simplicity in file format to get
    # the times for each file
    s_key = "X_X_0_" + start_filekey + ".txt"
    e_key = "X_X_0_" + end_filekey + ".txt"
    # gesture vid path is something like
    # "gest-data/data/rock/keypoints_simple/1/"
    files = sorted(os.listdir(gesture_video_path))
    # find the file that has this specific time.. I don't think there can be a clash??
    m = [s for s in files if start_filekey in s]
    if len(m) != 1:
        print "panic!! wrong number of matching files:" + str(len(m))
        print start_filekey
        print m
        return
    all_gesture_keys = []
    i = files.index(m[0])
    print "starting at " + str(files[i])
    # start at index of first frame
    while is_within_time(s_key, e_key, files[i]):
        ## if this is one of the files we need,
        ## we're constructing a pd dataset
        # fn = files[i].split('.txt')[0]
        # write csv file -- but can't be creating files because then our index gets messed up.
        #txt_to_csv(files[i])
        # read in data from csv in form of x,y ==> 52 rows of datapoints
        #dat = pd.read_csv(str(BASE_PATH + fn + '.csv'))
        dat = extract_txt_data(files[i])
        ## fuck it let's just use a dict for now.
        all_gesture_keys.append({'x': list(dat['x']), 'y': list(dat['y'])})
        i+=1
        if i >= len(files):
            print "WARNING: GOING BEYOND KEYPOINT TIMES: " + str(files[i-1])
            break
    ## the -1 is a hack until I figure out why there's missing keypoint data
    ## in some of these.
    print("stopped at ", files[i-1])
    # this will look something like this
    # [
    #   {
    #       x: [45, 321, 43...]
    #       y: [732, 21, 78...]
    #   },
    #   {
    #       x: [...]
    #       y: [...]
    #   }
    # ]
    return all_gesture_keys

def analyze_gestures(video_base_path, timings_path):
    all_gesture_data = []
    timings = get_timings(timings_path)

    for phase in timings['phrases']:
        p = phase['phase']
        vid_path = video_base_path + str(p['video_fn'].split('.')[0]) + '/'
        # TODO  OH GOD PLEASE UPDATE THIS
        BASE_PATH = vid_path
        start = p['start_seconds']
        end = p['end_seconds']
        specific_gesture_dat = get_keyframes_per_gesture(vid_path, start, end)
        all_gesture_data.append(specific_gesture_dat)
    return


def get_timings(timings_path):
    with open(timings_path) as f:
        timings = json.load(f)
    return timings

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
    get_speaker_gesture_keypoints = analyze_gestures(video_path, timings_path)

    print len(get_speaker_gesture_keypoints)


## TODO write unit tests for those bad boys up above
