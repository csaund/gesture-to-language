print "loading modules"
import argparse
from tqdm import tqdm
import subprocess
import os
import pandas as pd
import json
from common_helpers import *

## store it in the CLOUD
devKey = str(open("/Users/carolynsaund/devKey", "r").read()).strip()

## don't think this is necessary...?
# from apiclient.discovery import build
# service = build('language', 'v1', developerKey=devKey)
# collection = service.documents()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/carolynsaund/google-creds.json"

## read data from gesture data area
## transform into json I suppose
def convert_time_to_seconds(time):
    # might not work with longer ones?
    intervals = time.split(':')
    # [hours, minutes, seconds.ms]
    seconds = (float(intervals[0]) * 3600) + (float(intervals[1]) * 60) + float(intervals[2])
    return seconds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-base_path', '--base_path', help='base folder path of dataset', required=True)
    parser.add_argument('-speaker', '--speaker', default='optionally, run only on specific speaker', required=False)

    args = parser.parse_args()
    SGG = SpeakerGestureGetter(args.base_path, args.speaker)
    gs = SGG.perform_gesture_analysis()
