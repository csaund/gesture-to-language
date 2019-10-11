#!/usr/bin/env python
print "importing libs"
import argparse
from analyze_frames import analyze_gestures




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-base_path', '--base_path', help='base folder path of dataset', required=True)
    parser.add_argument('-output_path', '--output_path', default='output directory to save wav files', required=True)
    parser.add_argument('-speaker', '--speaker', default='optionally, run only on specific speaker', required=False)
    # --base_path /Users/carolynsaund/github/gest-data/data --speaker rock

    video_path = args.base_path + '/' + args.speaker + '/keypoints_simple/'
    timings_path = args.base_path + '/' + args.speaker + '/timings.json'

    print "processing data"
    all_speaker_gesture_keypoints = analyze_gestures(video_path, timings_path)
