#!/usr/bin/env pythons
print "importing for libs SpeakerGestureGetter"
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
from tqdm import tqdm
devKey = str(open("%s/devKey" % os.getenv("HOME"), "r").read()).strip()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "%s/google-creds.json" % os.getenv("HOME")
from common_helpers import *


AGD_BUCKET = "all_gesture_data"

## given a speaker name and path to their keypoints and gesture timings,
## loads all the gesture data to be analyzed.
class SpeakerGestureGetter():
    def __init__(self, base_path, speaker):
        self.video_path = "%s/%s/keypoints_simple/" % (base_path, speaker)
        self.timings_path = "%s/%s/timings.json" % (base_path, speaker)
        self.speaker = speaker
        self.base_path = base_path
        # self.all_gesture_data = self.analyze_gestures(self.video_path, self.timings_path)

    def perform_gesture_analysis(self, force_upload=False):
        self.force_upload = force_upload
        self.all_gesture_data = self.analyze_gestures(self.video_path, self.timings_path)
        return self.all_gesture_data

    def get_all_speaker_gesture_keypoints(self):
        return self.all_gesture_data

    # takes number of seconds (408.908909) and converts to something like
    # MM_S.SS
    # that is searchable in the folder.
    def _get_filekey(self, t):
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
    def get_keyframes_per_gesture(self, gesture_video_path, start_time, end_time):
        start_filekey = str(self._get_filekey(start_time))
        end_filekey = str(self._get_filekey(end_time))
        s_key = "X_X_0_" + start_filekey + ".txt"  # I am an absolute LUG. This is for simplicity in file format to get times for each file
        e_key = "X_X_0_" + end_filekey + ".txt"
        files = sorted(os.listdir(gesture_video_path), key=timestring_to_int)
        m = [s for s in files if start_filekey in s]
        if len(m) != 1:
            print("panic!! wrong number of matching files: %s" % str(len(m)))
            print(start_filekey)
            print(m)
            return
        all_gesture_keys = []
        i = files.index(m[0])
        #  print("starting at %s" % files[i])   # start at index of first frame
        l = len(files)
        while i < len(files) and is_within_time(s_key, e_key, files[i]):
            dat = extract_txt_data(gesture_video_path, files[i])
            all_gesture_keys.append(dat) # TODO change this to pd
            i+=1
        return all_gesture_keys



    ########################################################
    ################ All the plotting stuff ################
    ########################################################

    ## flip so instead of format like
    # [t1, t2, t3], [t1`, t2`, t3`], [t1``, t2``, t3``]
    # it's in the format of
    # [t1, t1`, t1``], [t2, t2`, t2``], [t3, t3`, t3``]
    def arrange_data_by_time(self, dat_vector):
        flipped_dat = []
        for i in range(len(dat_vector[0])):
            a = []
            for d in dat_vector:
                a.append(d[i])
            flipped_dat.append(a)
        return flipped_dat


    def plot_both_gesture_coords(self, gesture):
        plt.subplot(1,2,1)
        plot_coords('x', gesture)
        plt.subplot(1,2,2)
        plot_coords('y', gesture)
        plt.title = 'xy coordinates for gesture %s' % gesture['id']
        plt.savefig('%s.png' % gesture['id'])
        plt.show()
        return


    def plot_coords(self, x_y, gesture):
        coords = coords = [d[x_y] for d in gesture['keyframes']]
        fc = arrange_data_by_time(coords)
        for v in fc:
            plt.plot(range(0, len(fc[0])), v)
        plt.xlabel("frame")
        plt.ylabel("%s pixel position" % x_y)
        plt.title = '%s coordinates for gesture %s' % (x_y, gesture['id'])


    def analyze_gestures(self, video_base_path, timings_path):
        if(self.force_upload):
            return self.analyze_and_upload(video_base_path, timings_path)
        try:
            all_gesture_data = download_blob(AGD_BUCKET, "%s_agd.json" % self.speaker)
            return all_gesture_data
        except:
            self.analyze_and_upload(video_base_path, timings_path)

    def analyze_and_upload(self, video_base_path, timings_path):
        all_gesture_data = []
        timings = self.get_timings(timings_path)
        phrases = timings['phrases']
        ## TODO try this out...?
        # all_gesture_data = [self.get_data_per_gesture(g, video_base_path + str(g['phase']['video_fn'].split('.')[0]) + '/') for g in phrases]
        l = len(timings['phrases'])
        print "analyzing %s gestures" % l
        i = 0
        for phase in tqdm(timings['phrases']):
            # print i
            # i += 1
            p = phase['phase']
            vid_path = video_base_path + str(p['video_fn'].split('.')[0]) + '/'
            start = p['start_seconds']
            end = p['end_seconds']
            specific_gesture_dat = {'id': phase['id']}
            specific_gesture_dat['keyframes'] = self.get_keyframes_per_gesture(vid_path, start, end)
            all_gesture_data.append(specific_gesture_dat)

        upload_object(AGD_BUCKET, all_gesture_data, "%s_agd.json" % self.speaker)
        return all_gesture_data

    ## TODO
    ## my attempt at speeding things up a bit.
    def get_data_per_gesture(self, gest, vid_path):
        p = gest['phase']
        return {
            'id': p['id'],
            'keyframes': self.get_keyframes_per_gesture(vid_path, p['start_seconds'], p['end_seconds'])
        }

    # takes [id1, id2, id3] and saves
    # the plot images of those ids
    def save_gesture_plots(self, gesture_ids, all_gestures):
        for i in gesture_ids:
            g = self.get_gesture_by_id(i, all_gestures)
            self.plot_both_gesture_coords(i)
        return

    def plot_dist_of_num_frames_by_gesture(self, all_gestures):
        num_frames = []
        for g in all_gestures:
            num_frames.append(len(g['keyframes']))
        num_bins = 5
        n, bins, patches = plt.hist(num_frames, bins=[0,20,100,200,400,2500], facecolor='blue', alpha=0.5)
        plt.title = "number of frames per gesture"
        plt.show()


    def get_gesture_by_id(self, d_id, all_speaker_gesture_data=None):
        agd = all_speaker_gesture_data if all_speaker_gesture_data else self.all_gesture_data
        dat = [d for d in all_speaker_gesture_data if d['id'] == d_id]
        # because this returns list of matching items, and only one item will match,
        # we just take the first element and use that.
        return dat[0]

    def get_timings(self, timings_path):
        with open(timings_path) as f:
            timings = json.load(f)
        return timings

    def is_within_time_ez(self, s, e, q):
        if q < s or q > e:
            return False
        return True

    def get_single_gesture_from_timings(self, gesture_id, video_base_path, timings_path):
        gesture_data = {}
        timings = get_data_from_path(timings_path)
        for phase in timings['phrases']:
            p = phase['phase']
            if phase['id'] == gesture_id:
                start = p['start_seconds']
                end = p['end_seconds']
                vid_path = video_base_path + str(p['video_fn'].split('.')[0]) + '/'
                gesture_data['id'] = phase['id']
                gesture_data['keyframes'] = self.get_keyframes_per_gesture(vid_path, start, end)
        return gesture_data

    # takes time in MM_SS.MS
    def get_gesture_ids_from_video_and_timing(self, video_fn_id, time_in_mm_ss, timings_path):
        t = get_data_from_path(timings_path)
        candidates = []
        # transform a reasonable time to a dumb time.
        t_to_trans = "X_X_0_%s.txt" % time_in_mm_ss
        time_in_seconds = timestring_to_int(t_to_trans)
        all_video_gestures = [p for p in t['phrases'] if p['phase']['video_fn'] == video_fn_id]
        for gesture in all_video_gestures:
            p = gesture['phase']
            if is_within_time_ez(p['start_seconds'], p['end_seconds'], time_in_seconds):
                candidates.append(gesture['id'])
        return candidates

    # get number of frames for a particular gesture by gesture id
    def get_num_frames(self, gesture_id):
        g = get_gesture_by_id(gesture_id, self.all_gesture_data)
        return len(g['keyframes'])

    ## helper, so can use all these random vars I have floating around I guess
    def get_plot_by_video_time(self, time_in_mm_ss,
                              video_fn_id="2._History_of_Rock_-_Radio_and_Regional_vs_National_Audiences-diqfvS-VlzI.mp4"):
        ids = self.get_gesture_ids_from_video_and_timing(video_fn_id, time_in_mm_ss, self.timings_path)
        if len(ids) > 1:
            print "more than one gesture associated with this time, using gesture %s" % ids[0]
        elif len(ids) == 0:
            print "no gestures found for this time"
            return
        i = ids[0]
        print i
        gest = self.get_single_gesture_from_timings(i, self.video_path, self.timings_path)
        self.plot_both_gesture_coords(gest)
        return

    # gest_types include
    # BEATS
    # WISH_WASH
    # THIS_THAT
    # THIS_THAT_SPATIAL_MAP
    # TIME_SIDE_SWIPE
    # TIME_FORWARD_SWIPE
    # MIXING
    # BIG_OPENING
    # PALM_DOWN_BLANKET
    def save_id_to_video_type(self, gest_id, gest_type, labeled_path="/Users/carolynsaund/github/gesture-to-language/manually_labelled_gestures.json"):
        d = {
            "id": gest_id,
            "cat": gest_type
        }
        with open(labeled_path) as f:
            labeled_gestures = json.load(f)
            labeled_gestures['gestures'].append(d)
            with open(labeled_path, 'w') as f:
                json.dump(labeled_gestures, f, indent=4)
        return

## TODO write unit tests for those bad boys up above
