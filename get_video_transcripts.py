#!/usr/bin/env python
import os
import argparse
import io
import subprocess
import json
import copy
import uuid

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
# TODO make this take the whole video and transcribe all at once.
def transcribe_videos(vid_base_path, transcript_path):
    """Transcribe the given audio file synchronously and output the word time
    offsets."""
    from google.cloud import speech
    from google.cloud.speech import enums
    from google.cloud.speech import types
    client = speech.SpeechClient()

    all_video_files = os.listdir(vid_base_path)
    print "seeing all files:"
    print all_video_files

    ## TODO: trim video so that it fits within google's size limits

    for video_file in all_video_files:
        print "now on file " + video_file
        print
        vid_name = video_file.split(".mp4")[0]
        input_vid_path = vid_base_path + '/' + video_file
        output_audio_path = transcript_path + '/' + vid_name + '.wav'

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

        output_data = {}
        output_data['words'] = []
        for result in response.results:
            alternative = result.alternatives[0]
            output_data["transcript"] = str(alternative.transcript)

            ## TODO fix this so we dont' have to do all this setup the first time around...
            ## go through and add specific transcripts to each gesture based on timings
            gesture_index = 0
            current_gesture = gesture_phrase['gestures'][0]
            current_gesture['transcript'] = ""
            gesture_phrase_start = gesture_phrase['phase']['start_seconds']
            current_gesture['id'] = str(uuid.uuid1())
            for i in range(len(alternative.words)):
                word_info = alternative.words[i]

                ## nanos is super f'ed up in these responses? like they go up and down a bunch?
                ## let's just try with seconds and see where we get.
                # word_start = word_info.start_time.nanos *  1e-9
                # word_end = word_info.start_time.nanos *  1e-9
                word_start = word_info.start_time.seconds
                word_end = word_info.start_time.seconds
                word = {
                    "word_info": word_info,
                    "word_start": word_start,
                    "word_end": word_end
                }
                outputdata['words'].append(word)

        output_transcript_path = transcript_path + '/' + vid_name + '.json'
        fn = output_transcript_path
        with open(fn, 'w') as f:
            json.dump(output_data, f, indent=4)
        f.close()


def create_video_subdir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        print ("Creation of the directory %s failed" % dir_path)
    else:
        print ("Successfully created the directory %s " % dir_path)


def get_video_transcript(video_path, transcript_path):
    create_video_subdir(transcript_path)
    transcribe_videos(video_path, transcript_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-base_path', '--base_path', help='base folder path of dataset', required=True)
    #parser.add_argument('-output_path', '--output_path', default='output directory to save wav files', required=True)
    parser.add_argument('-speaker', '--speaker', default='optionally, run only on specific speaker', required=False)

    args = parser.parse_args()

    vid_base_path = args.base_path + '/' + args.speaker + '/videos'
    transcript_path = args.base_path + '/' + args.speaker + '/transcripts'
    # TODO make other script output timings to this place


    ## TODO a better way to do this is get the whole video transcript, then using the gesture timings,
    ## match the timing of the transcript to the timing of the gesture. That is strictly a better way
    ## to do all of this. 
    get_video_transcript(vid_base_path, transcript_path)
