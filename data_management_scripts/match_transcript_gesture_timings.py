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
from tqdm import tqdm

devKey = str(open("%s/devKey" % os.getenv("HOME"), "r").read()).strip()
from apiclient.discovery import build
service = build('language', 'v1', developerKey=devKey)
collection = service.documents()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "%s/google-creds.json" % os.getenv("HOME")
client = bigquery.Client()
dataset_id = 'my_dataset'


gesture_timings_bucket = "speaker_timings"
transcript_bucket = "audio_transcript_buckets_1"
full_bucket = "full_timings_with_transcript_bucket"

## TODO get full transcript and work across syntacic units

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
    download_blob(transcript_bucket, video_name.replace(".mp4", ".json").replace(".mkv", ".json").replace(".webm", ".json"), outfile_path)
    transcript = read_data(outfile_path)
    #os.remove(outfile_path)
    return transcript

def match_transcript_to_timing(timings):
    phrases = timings['phrases']

    # break up timings by transcripts (all videos)
    # then sort by start time because the transcripts are already sorted by start time.
    all_videos = list(set([p['phase']['video_fn'] for p in phrases]))

    # avoid loading all the transcripts into memory at once
    for v in tqdm(all_videos):
        trans = ""
        try:
            trans = get_video_transcript(v)
        except:
            print "Skipping %s." % v
            continue

        if len(trans) == 0:
            print "Something is going wrong. Aborting video transcript"
            continue

        all_words = sorted(flatten([t['words'] for t in trans]), key=word_sort)
        # word_iter = iter(all_words)
        # current_word = next(word_iter)
        # can't use an iterator here because we need to include some words in
        # multiple gestures, aka use the words for the gestures that don't have
        # any words...
        len_words = len(all_words)
        current_word = all_words[0]
        word_i = 0
        # break up our dict based on the video transcript.
        # get all the items from our phrases with this video fn
        gestures = [x for x in phrases if x['phase']['video_fn'] == v]
        gestures.sort(key=sort_start_time)
        print "Getting gestures for video %s" % v
        for g in gestures:
            gesture_words = []
            p = g['phase']
            end = p['end_seconds']
            while current_word['word_start'] <= end and word_i < len(all_words):
                current_word = all_words[word_i]
                gesture_words.append(current_word)
                word_i += 1

            ## python shallow copies OH YEAHHHH
            # if we don't have any words, just take the one before and the next 3.
            g['words'] = gesture_words if gesture_words else get_word_range(word_i, all_words)
            words = [g['word'] for g in g['words']]
            p['transcript'] = ' '.join(words)


    # consistency...
    nt = {'phrases': sorted(timings['phrases'], key=lambda x: (x['phase']['video_fn'], x['phase']['start_seconds']))}
    return nt

def get_word_range(start_range, words):
    BEGIN_RANGE = -1
    END_RANGE = 3
    if (start_range + 3) >= len(words):
        return(words[-(END_RANGE+1):])
    else:
        return(words[(start_range + BEGIN_RANGE):(start_range + END_RANGE)])


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
    print "uploading transcript timings"
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
    args = parser.parse_args()

    add_transcript_data(args.speaker, args.base_path)

# from match_transcript_gesture_timings import *
# sudo python match_transcript_gesture_timings.py --base_path /Users/carolynsaund/github/gest-data/data --speaker rock
