#!/usr/bin/env pythons
import json
import os
import argparse
from common_helpers import *

devKey = str(open("/Users/carolynsaund/devKey", "r").read()).strip()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/carolynsaund/google-creds.json"

from google.cloud import storage


## the simplest way to do this is to create a new "conglomerate" speaker who is composed of all the other speakers.
transcript_bucket = "audio_transcript_buckets_1"
FULL_TRANSCRIPT_BUCKET = "full_timings_with_transcript_bucket"     ## Used in SentenceClusterer
AGD_BUCKET = "all_gesture_data"     ## Used to load gestures
timings_bucket = "speaker_timings"

def concat_timings_transcript_id_keyframes(speakers):
    agd = {}
    agd_phrases = []
    id_keyframes = []
    for s in speakers:
        agd_phrases.append(get_data_from_blob(FULL_TRANSCRIPT_BUCKET, "%s_timings_with_transcript.json" % s)['phrases'])
        id_keyframes.append(get_data_from_blob(AGD_BUCKET), "%s_agd.json" % s)
    agd = {'phrases': agd_phrases}

    upload_object(FULL_TRANSCRIPT_BUCKET, agd, "conglomerate_timings_with_transcript.json")
    upload_object(AGD_BUCKET, id_keyframes, "conglomerate_agd.json")


def conglomerate_speakers(base_path, speakers):
    concat_timings_transcript_id_keyframes(speakers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-base_path', '--base_path', help='base folder path of dataset', required=True)
    parser.add_argument('-speakers', '--speakers', default='optionally, run only on specific speaker', nargs='+')
    args = parser.parse_args()

    #vid_base_path = args.base_path + '/' + args.speaker + '/videos'
    #transcript_path = args.base_path + '/' + args.speaker + '/transcripts'
    conglomerate_speakers(args.base_path, args.speakers)
    print args.speakers
