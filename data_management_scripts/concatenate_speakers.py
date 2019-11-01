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
full_timings_transcript_bucket = "full_timings_with_transcript_bucket"     ## Used in SentenceClusterer
agd_bucket = "all_gesture_data"     ## Used to load gestures
timings_bucket = "speaker_timings"

def get_timings(speaker):




def concat_timings(speakers):
    agd = {}
    agd_phrases = []
    for s in speakers:
        agd_phrases.append(get_data_from_blob(full_transcript_bucket, "%s_timings_with_transcript.json" % s)['phrases'])

    agd = {'phrases': agd_phrases}
    upload_object(full_transcript_bucket, agd)



def conglomerate_speakers(base_path, speakers):
    concat_timings(base_path, speakers)
    concat_full_transcripts(base_path, speakers, "conglomerate_timings_with_transcript.json")


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
