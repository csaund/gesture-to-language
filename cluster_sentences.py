#!/usr/bin/env python
import os
import argparse
import io
import subprocess
import nltk




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'path', help='Long mp4 file to be segmented into gestures')
    args = parser.parse_args()

    vid_path = args.path
    filename_base = vid_path.split('/')[-1].split('.')[-2]

    create_video_subdir(filename_base)

    ## TEMP right now just hard coded for testing
    ## TODO: make this actually segment by gesture
    ## 12 Sept 2019
    gesture_clips = make_clip_timings()
    segment_and_extract(filename_base, vid_path, gesture_clips)
