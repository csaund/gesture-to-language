print "loading modules for get_keyframes_for_gesture"
import argparse
import os
import sys
sys.path.append('../')
from SpeakerGestureGetter import *

## store it in the CLOUD

devKeyPath = "%s/devKey" % os.getenv("HOME")
devKey = str(open(devKeyPath, "r").read()).strip()

## don't think this is necessary...?
# from apiclient.discovery import build
# service = build('language', 'v1', developerKey=devKey)
# collection = service.documents()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "%s/google-creds.json" % os.getenv("HOME")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-base_path', '--base_path', help='base folder path of dataset', required=True)
    parser.add_argument('-speaker', '--speaker', default='optionally, run only on specific speaker', required=False)
    parser.add_argument('-force', '--force_upload', default=False)

    args = parser.parse_args()
    SGG = SpeakerGestureGetter(args.base_path, args.speaker)
    gs = SGG.perform_gesture_analysis(args.force_upload)
