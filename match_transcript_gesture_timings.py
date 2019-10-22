#!/usr/bin/env python
print "importing libs"
import os
import io
import argparse
import subprocess
import json
import argparse
from google.cloud import storage
from google.cloud import bigquery
from common_helpers import *

devKey = str(open("/Users/carolynsaund/devKey", "r").read()).strip()
from apiclient.discovery import build
service = build('language', 'v1', developerKey=devKey)
collection = service.documents()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/carolynsaund/google-creds.json"
client = bigquery.Client()
dataset_id = 'my_dataset'


gesture_timings_bucket = "speaker_timings"
transcript_bucket = "audio_transcript_buckets_1"
full_bucket = "full_timings_with_transcript_bucket"

def get_speaker_timings(speaker):
    print "getting timings for %s" % speaker
    outfile_path = "tmp.json"
    touch(outfile_path)
    download_blob(gesture_timings_bucket, speaker + "_timings.json", outfile_path)
    timings = read_data(outfile_path)
    os.remove(outfile_path)
    return timings

def get_video_transcript(video_name):
    print "getting transcript for %s" % video_name
    outfile_path = "tmp.json"
    touch(outfile_path)
    download_blob(transcript_bucket, video_name.replace(".mp4", ".json"), outfile_path)
    transcript = read_data(outfile_path)
    os.remove(outfile_path)
    return transcript

def match_transcript_to_timing(timings):
    phrases = timings['phrases']

    # break up timings by transcripts (all videos)
    # then sort by start time because the transcripts are already sorted by start time.
    all_videos = list(set([p['phase']['video_fn'] for p in phrases]))

    # avoid loading all the transcripts into memory at once
    for v in all_videos:
        trans = ""
        try:
            trans = get_video_transcript(v)
        except:
            print "Skipping %s." % v
            continue

        all_words = sorted(flatten([t['words'] for t in trans]), key=word_sort)
        word_iter = iter(all_words)
        current_word = next(word_iter)

        # break up our dict based on the video transcript.
        # get all the items from our phrases with this video fn
        gestures = [x for x in phrases if x['phase']['video_fn'] == v]
        gestures.sort(key=sort_start_time)
        for g in gestures:
            gesture_words = []
            gesture_transcript = ""
            p = g['phase']
            end = p['end_seconds']
            while current_word['word_start'] <= end:
                gesture_words.append(current_word)
                gesture_transcript += " " + current_word['word']
                try:
                    current_word = next(word_iter)
                except StopIteration:
                    break

            ## python shallow copies OH YEAHHHH
            g['words'] = gesture_words
            p['transcript'] = gesture_transcript

    # consistency...
    nt = {'phrases': sorted(timings['phrases'], key=lambda x: (x['phase']['video_fn'], x['phase']['start_seconds']))}
    return nt

def word_sort(x):
    return x['word_start']

def write_transcript(transcript, transcript_path):
    with open(transcript_path, 'w') as f:
        json.dump(transcript, f, indent=4)
    f.close()

def flatten(x):
    return [val for sublist in x for val in sublist]

def sort_start_time(x):
    return x['phase']['start_seconds']

def add_transcript_data(speaker, base_path):
    timings = get_speaker_timings(speaker)
    timings_with_transcript = match_transcript_to_timing(timings)
    filename = "%s_timings_with_transcript.json" % speaker
    output_path = "%s/%s/%s" % (base_path, speaker, filename)
    write_transcript(timings_with_transcript, output_path)
    upload_blob(full_bucket, output_path, filename)

## take gesture timings for a speaker
## take video transcripts for that speaker
## matches them up
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-base_path', '--base_path', help='base folder path of dataset', required=True)
    parser.add_argument('-speaker', '--speaker', default='optionally, run only on specific speaker', required=False)
    # --base_path /Users/carolynsaund/github/gest-data/data --speaker rock
    args = parser.parse_args()

    add_transcript_data(args.speaker, args.base_path)

# from match_transcript_gesture_timings import *
