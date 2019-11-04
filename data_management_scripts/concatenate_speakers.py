#!/usr/bin/env pythons
import json
import os
import argparse
from common_helpers import *
from tqdm import tqdm

devKey = str(open("%s/devKey" % os.getenv("HOME"), "r").read()).strip()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "%s/google-creds.json" % os.getenv("HOME")

from google.cloud import storage


## the simplest way to do this is to create a new "conglomerate" speaker who is composed of all the other speakers.
transcript_bucket = "audio_transcript_buckets_1"
FULL_TRANSCRIPT_BUCKET = "full_timings_with_transcript_bucket"     ## Used in SentenceClusterer
AGD_BUCKET = "all_gesture_data"     ## Used to load gestures
timings_bucket = "speaker_timings"

def concat_timings_transcript_id_keyframes(speakers):
    trans_phrases = []  #transcripts seem to be fine...
    id_keyframes = [] # why are keyframes in wrong format??
    # I know, it's because I am a dummy
    for s in tqdm(speakers):
        print "Getting data and transcript for %s" % s
        trans_data = get_data_from_blob(FULL_TRANSCRIPT_BUCKET, "%s_timings_with_transcript.json" % s)['phrases']
        agd_data = get_data_from_blob(AGD_BUCKET, "%s_agd.json" % s)

        trans_phrases = trans_phrases + trans_data
        id_keyframes = id_keyframes + agd_data

    all_trans = {'phrases': trans_phrases}

    print "Uploading full transcript"
    upload_object(FULL_TRANSCRIPT_BUCKET, all_trans, "conglomerate_timings_with_transcript.json")
    print "Uploading full gesture data"
    upload_object(AGD_BUCKET, id_keyframes, "conglomerate_agd.json")


def conglomerate_speakers(speakers):
    concat_timings_transcript_id_keyframes(speakers)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-speakers', '--speakers', default='optionally, run only on specific speaker', nargs='+')
    args = parser.parse_args()

    #vid_base_path = args.base_path + '/' + args.speaker + '/videos'
    #transcript_path = args.base_path + '/' + args.speaker + '/transcripts'
    conglomerate_speakers(args.speakers)
