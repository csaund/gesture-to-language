#!/usr/bin/env python
import os
import argparse
import io
import subprocess
import json
import copy

# from moviepy.editor import VideoFileClip, AudioFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from oauth2client.client import GoogleCredentials

devKey = str(open("/Users/carolynsaund/devKey", "r").read()).strip()

from apiclient.discovery import build
service = build('language', 'v1', developerKey=devKey)
collection = service.documents()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/carolynsaund/google-creds.json"


#   "phrases":
#   [
#       {
#         "id": 1,
#         "phase": {
#             "start_seconds": 363,
#             "end_seconds": 378.1
#         },
#         "gestures": [
#             {
#               "start_seconds": 364,
#               "end_seconds": 365.5
#             },
#             {
#               "start_seconds": 365.6,
#               "end_seconds": 366
#             },
#         ]
#     },
#     ...
# ]
# writes out to filename_base/filename_base-id.json
# with addition of words that match the gesture phrase.
# gesture_clip_timings is the dict that we want to append to.
def transcribe_files(filename_base, gesture_clip_timings):
    """Transcribe the given audio file synchronously and output the word time
    offsets."""
    from google.cloud import speech
    from google.cloud.speech import enums
    from google.cloud.speech import types
    client = speech.SpeechClient()
    dir_path = "./" + filename_base

    output_data = copy.deepcopy(gesture_clip_timings)

    for gesture_phrase in output_data:
        print
        print "NEW GESTURE"
        input_vid_path = dir_path + '/' + filename_base + '_' + str(gesture_phrase['id']) + '.mp4'
        output_audio_path = dir_path + '/' + filename_base + '_' + str(gesture_phrase['id']) + '.wav'
        # TODO there is a better way of iterating thru all items in a dir probably

        command = ("ffmpeg -i %s -ab 160k -ac 2 -ar 48000 -vn %s" % (input_vid_path, output_audio_path))
        subprocess.call(command, shell=True)

        with io.open(output_audio_path, 'rb') as audio_file:
            content = audio_file.read()

        audio = types.RecognitionAudio(content=content)
        config = types.RecognitionConfig(
            encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=48000,
            audio_channel_count=2,
            language_code='en-US',
            enable_word_time_offsets=True)

        response = client.recognize(config, audio)

        for result in response.results:
            alternative = result.alternatives[0]
            gesture_phrase['phase']['transcript'] = str(alternative.transcript)

            ## go through and add specific transcripts to each gesture based on timings
            gesture_index = 0
            current_gesture = gesture_phrase['gestures'][0]
            current_gesture['transcript'] = ""
            gesture_phrase_start = gesture_phrase['phase']['start_seconds']
            for i in range(len(alternative.words)):
                word_info = alternative.words[i]

                ## nanos is super f'ed up in these responses? like they go up and down a bunch?
                ## let's just try with seconds and see where we get.
                # word_start = word_info.start_time.nanos *  1e-9
                # word_end = word_info.start_time.nanos *  1e-9
                word_start = word_info.start_time.seconds
                word_end = word_info.start_time.seconds
                # we're still in the same gesture phrase
                if (word_start + gesture_phrase_start) < current_gesture['end_seconds']:
                    current_gesture['transcript'] += " " + word_info.word
                else:
                    gesture_index += 1
                    current_gesture = gesture_phrase['gestures'][gesture_index]
                    current_gesture['transcript'] = word_info.word
            ## only do this for first alternative.
            break

    output_transcript_path = dir_path + '/' + filename_base + '_transcripts' + '.json'
    fn = output_transcript_path
    with open(fn, 'w') as f:
        json.dump(output_data, f, indent=4)
    f.close()


## TODO calculate the timestamps automatically, or take them as an arg
# expects timings file to exist as well
def make_clip_timings(filename_base):
    timings_file = filename_base + '-timings.json'
    with open(timings_file) as f:
        timings = json.load(f)
    return timings



## assumes video subfolder is already created
# get timings as list of dicts with shape:
#   "phrases":
#   [
#       {
#         "id": 1,
#         "phase": {
#             "start_seconds": 363,
#             "end_seconds": 378.1
#         },
#         "gestures": [
#             {
#               "start_seconds": 364,
#               "end_seconds": 365.5
#             },
#             {
#               "start_seconds": 365.6,
#               "end_seconds": 366
#             },
#         ]
#     },
#     ...
# ]
def segment_video(filename_base, video_path, gesture_clip_timings):
    dir_path = "./" + filename_base
    for gesture_phase in gesture_clip_timings:
        output_vid_path = dir_path + '/' + filename_base + '_' + str(gesture_phase['id']) + '.mp4'
        ffmpeg_extract_subclip(video_path, gesture_phase['phase']['start_seconds'], gesture_phase['phase']['end_seconds'], targetname=output_vid_path)
    return

def segment_and_extract(filename_base, video_path, gesture_clip_timings):
    segment_video(filename_base, video_path, gesture_clip_timings)
    transcribe_files(filename_base, gesture_clip_timings)

def create_video_subdir(filename_base):
    dir_path = "./" + filename_base
    try:
        os.mkdir(dir_path)
    except OSError:
        print ("Creation of the directory %s failed" % dir_path)
    else:
        print ("Successfully created the directory %s " % dir_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'path', help='Long mp4 file to be segmented into gestures')
    args = parser.parse_args()

    vid_path = args.path
    filename_base = vid_path.split('/')[-1].split('.')[-2]

    create_video_subdir(filename_base)

    ## TEMP right now just hard coded for testing
    ## TODO: make this actually segment by gesture
    ## 12 Sept 2019
    gesture_clips = make_clip_timings(filename_base)
    segment_and_extract(filename_base, vid_path, gesture_clips["phrases"])
