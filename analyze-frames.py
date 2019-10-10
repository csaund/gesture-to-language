#!/usr/bin/env pythons
print "importing libs"
import argparse
import pandas as pd
import itertools
import csv
import glob, os


# takes start time in form of X_X_IND_m_s.txt, end time in form of X_X_IND_m_s.txt
# and question time in form of X_X_IND_m_s.txt
# returns whether or not question time is between start and end times
def is_within_time(start_time, end_time, question_time):
    return True

# takes fp to text file, returns dataframe of
# csv version of that text file
# however it takes the x y and zips them together to make
# a big vector of x,y pairs.
def txt_to_csv(filepath):
    fn = filepath.split('.txt')[0]
    inf = fn + '.txt'
    outf = fn + '.csv'
    with open(inf, 'r') as in_file:
        lines = in_file.read().splitlines()
        stripped = [line.replace(","," ").split() for line in lines]
        zipped = zip(stripped[1], stripped[2])
        with open(outf, 'w') as out_file:
            writer = csv.writer(out_file)
            writer.writerow(('x', 'y'))
            i = 0
            for z in zipped:
                row = (str(z[0]), str(z[1]))
                print row
                writer.writerow(row)
                i += 1

def get_keyframes_per_gesture(gesture_video_path, start_time, end_time):
    start_keyframe_min = start_time / 60
    start_keyframe_sec = round(start_time - (start_keyframe_min * 60), 6)
    ## in the file, there will be this.
    filekey = str(start_keyframe_min) + '_' + str(start_keyframe_sec)
    # gesture vid path is something like
    # "gest-data/data/rock/keypoints_simple/1/"
    files = sorted(os.listdir(gesture_video_path))
    # find the file that has this specific time.. I don't think there can be a clash??
    m = [s for s in files if filekey in s]
    if len(m) != 1:
        print "panic!! wrong number of matching files:" + len(m)
        exit(1)

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
    all_gesture_keys = []
    i = files.index(m[0])
    # start at index of first frame
    while is_within_time(files[i]):
        ## if this is one of the files we need,
        ## we're constructing a pd dataset
        fn = files[i].split('.txt')[0]
        # write csv file
        txt_to_csv(files[i])
        # read in data from csv in form of x,y ==> 52 rows of datapoints
        dat = pd.read_csv(str(fn + '.csv'))
        ## fuck it let's just use a dict for now.
        all_gesture_keys.append({'x': list(dat['x']), 'y': list(dat['y'])})
        i+=1
    return

def analyze_gesture()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-base_path', '--base_path', help='base folder path of dataset', required=True)
    #parser.add_argument('-output_path', '--output_path', default='output directory to save wav files', required=True)
    parser.add_argument('-speaker', '--speaker', default='optionally, run only on specific speaker', required=False)
    parser.add_argument('-frames', '--frames', default='path to frames file', required=False)

    args = parser.parse_args()
    timings_path = args.base_path + '/' + args.speaker + '/timings.json'

    print "reading csv from " + args.frames
    ## 7.8M frames
    frames = pd.read_csv(args.frames)

    frames.shape
