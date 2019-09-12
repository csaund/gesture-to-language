#!/usr/bin/env python
import os
import argparse
import io
import subprocess
import json

# from moviepy.editor import VideoFileClip, AudioFileClip
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from oauth2client.client import GoogleCredentials

devKey = str(open("/Users/carolynsaund/devKey", "r").read()).strip()

from apiclient.discovery import build
service = build('language', 'v1', developerKey=devKey)
collection = service.documents()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/carolynsaund/google-creds.json"

def transcribe_files(filename_base, gesture_clip_timings):
    """Transcribe the given audio file synchronously and output the word time
    offsets."""
    from google.cloud import speech
    from google.cloud.speech import enums
    from google.cloud.speech import types
    client = speech.SpeechClient()
    dir_path = "./" + filename_base

    for gesture in gesture_clip_timings:
        input_vid_path = dir_path + '/' + filename_base + '_' + str(gesture['id']) + '.mp4'
        output_audio_path = dir_path + '/' + filename_base + '_' + str(gesture['id']) + '.wav'
        output_transcript_path = dir_path + '/' + filename_base + '_' + str(gesture['id']) + '.json'
        # todo there is a better way of iterating thru all items in a dir probably

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

        fn = output_transcript_path
        transcript = {}
        with open(fn, 'w') as f:
            for result in response.results:
                alternative = result.alternatives[0]
                transcript['Transcript:'] = str(alternative.transcript)

                transcript['words'] = []
                for i in range(len(alternative.words)):
                    transcript['words'].append({})
                    word_info = alternative.words[i]
                    transcript['words'][i]['word'] = str(word_info.word)
                    transcript['words'][i]['start_time'] = word_info.start_time.nanos *  1e-9
                    transcript['words'][i]['end_time'] = word_info.end_time.nanos * 1e-9

            json.dump(dict(transcript), f)
        f.close()


## TODO calculate the timestamps automatically, or take them as an arg
# expects timings file to exist as well
def make_clip_timings(filename_base):
    timings_file = filename_base + '-timings.json'
    with open(timings_file) as f:
        timings = json.load(f)
    return timings


# get timings as list of dicts with shape:
#        {
#            id: 1,
#            start_seconds: 367.5,
#            end_seconds: 369.0
#        }
## assumes video subfolder is already created
def segment_video(filename_base, video_path, gesture_clip_timings):
    dir_path = "./" + filename_base
    for gesture in gesture_clip_timings:
        output_vid_path = dir_path + '/' + filename_base + '_' + str(gesture['id']) + '.mp4'
        ffmpeg_extract_subclip(video_path, gesture['start_seconds'], gesture['end_seconds'], targetname=output_vid_path)
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
    segment_and_extract(filename_base, vid_path, gesture_clips["gestures"])
