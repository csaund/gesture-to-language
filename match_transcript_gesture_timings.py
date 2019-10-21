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

devKey = str(open("/Users/carolynsaund/devKey", "r").read()).strip()
from apiclient.discovery import build
service = build('language', 'v1', developerKey=devKey)
collection = service.documents()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/carolynsaund/google-creds.json"
client = bigquery.Client()
dataset_id = 'my_dataset'


gesture_timings_bucket = "speaker_timings"
transcript_bucket = "audio_transcript_buckets_1"


def read_data(fp):
    with open(fp, 'r') as f:
        t = json.load(f)
        return t

def touch(path):
    with open(path, 'a'):
        os.utime(path, None)

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print('Blob {} downloaded to {}.'.format(
        source_blob_name,
        destination_file_name))

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
    # will loading all transcripts be too much for memory? oh well!

    # break up timings by transcripts (all videos)
    # then sort by start time because the transcripts are already sorted by start time.
    all_videos = list(set([p['phase']['video_fn'] for p in phrases]))

    # avoid loading all the transcripts into memory at once
    for v in all_videos:
        trans = ""
        try:
            trans = get_video_transcript(v)
        except:
            print "Couldn't get video transcript for %s. Skipping video" % v
            continue

        all_words = sorted(flatten([t['words'] for t in trans]), key=word_sort)
        word_iter = iter(all_words)
        current_word = next(word_iter)

        # break up our dict based on the video transcript.
        # get all the items from our phrases with this video fn
        gestures = [x for x in phrases if x['phase']['video_fn'] == v]
        gs = sorted(gestures, key=sort_start_time)
        for g in gs:
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

                # word_index += 1
                # print "word index: %s" % word_index
            ## python shallow copies OH YEAHHHH
            g['words'] = gesture_words
            g['transcript'] = gesture_transcript
    return timings

def word_sort(x):
    return x['word_start']

def flatten(x):
    return [val for sublist in x for val in sublist]

def sort_start_time(x):
    return x['phase']['start_seconds']

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

    timings = get_speaker_timings(args.speaker)
    timings_with_transcript = match_transcript_to_timing(timings)

# from match_transcript_gesture_timings import *
timings = get_speaker_timings("rock")
nt = match_transcript_to_timing(timings)
