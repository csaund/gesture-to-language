#!/usr/bin/env pythons
import os
import argparse
from common_helpers import *
from tqdm import tqdm

devKey = str(open("%s/devKey" % os.getenv("HOME"), "r").read()).strip()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "%s/google-creds.json" % os.getenv("HOME")

## the simplest way to do this is to create a new "conglomerate" speaker who is composed of all the other speakers.
transcript_bucket = "audio_transcript_buckets_1"
FULL_TRANSCRIPT_BUCKET = "full_timings_with_transcript_bucket"     ## Used in SentenceClusterer
AGD_BUCKET = "all_gesture_data"     ## Used to load gestures
timings_bucket = "speaker_timings"

def concat_timings_transcript_id_keyframes(speakers, n):
    name = "conglomerate"
    if n:
        name = name + "_under_%s" % n
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
    upload_object(FULL_TRANSCRIPT_BUCKET, all_trans, "%s_timings_with_transcript.json" % name)
    print "Uploading full gesture data"
    upload_object(AGD_BUCKET, id_keyframes, "%s_agd.json" % name)


def conglomerate_speakers(speakers, under_n):
    concat_timings_transcript_id_keyframes(speakers, under_n)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-speakers', '--speakers', default='optionally, run only on specific speaker', nargs='+')
    parser.add_argument('-under_n', '--under_n', help='keep the data under a certain number of words', default=10)
    args = parser.parse_args()

    conglomerate_speakers(args.speakers, args.under_n)
