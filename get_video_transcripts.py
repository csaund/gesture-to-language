#!/usr/bin/env python
print "importing libs"
import os
import argparse
import io
import subprocess
import json
import copy
import uuid

from pydub import AudioSegment
import wave
from google.cloud import storage
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
# import webapp2
# from google.appengine.api import app_identity

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from oauth2client.client import GoogleCredentials

devKey = str(open("/Users/carolynsaund/devKey", "r").read()).strip()
bucketname = "audio_bucket_rock_1"

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
# def transcribe_videos(vid_base_path, transcript_path):
#     """Transcribe the given audio file synchronously and output the word time
#     offsets."""
#     client = speech.SpeechClient()
#
#     ## have to use GCS for long audio files.
#     with io.open(output_audio_path, 'rb') as audio_file:
#         content = audio_file.read()
#
#     print "sending audio file to recognize"
#     audio = types.RecognitionAudio(content=content)
#     config = types.RecognitionConfig(
#         encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
#         sample_rate_hertz=48000,
#         audio_channel_count=2,
#         language_code='en-US',
#         enable_word_time_offsets=True)
#
#     response = client.recognize(config, audio)
#
#     print "got speech result"
#     output_data = {}
#     output_data['words'] = []
#     for result in response.results:
#         alternative = result.alternatives[0]
#         output_data["transcript"] = str(alternative.transcript)
#
#         ## TODO fix this so we dont' have to do all this setup the first time around...
#         ## go through and add specific transcripts to each gesture based on timings
#
#         ## TODO pass audio as gcs bucket uri
#         gesture_index = 0
#         current_gesture = gesture_phrase['gestures'][0]
#         current_gesture['transcript'] = ""
#         gesture_phrase_start = gesture_phrase['phase']['start_seconds']
#         current_gesture['id'] = str(uuid.uuid1())
#         for i in range(len(alternative.words)):
#             word_info = alternative.words[i]
#
#             ## nanos is super f'ed up in these responses? like they go up and down a bunch?
#             ## let's just try with seconds and see where we get.
#             # word_start = word_info.start_time.nanos *  1e-9
#             # word_end = word_info.start_time.nanos *  1e-9
#             word_start = word_info.start_time.seconds
#             word_end = word_info.start_time.seconds
#             word = {
#                 "word_info": word_info,
#                 "word_start": word_start,
#                 "word_end": word_end
#             }
#             outputdata['words'].append(word)
#
#     output_transcript_path = transcript_path + '/' + vid_name + '.json'
#     fn = output_transcript_path
#     with open(fn, 'w') as f:
#         json.dump(output_data, f, indent=4)
#     f.close()


#### NEW ####
def google_transcribe(audio_file_path):
    file_name = audio_file_path
    print "attempting to transcript file %s" % audio_file_path
    # frame_rate, channels = frame_rate_channel(file_name)
    #
    # if channels > 1:
    #     stereo_to_mono(file_name)
    bucket_name = bucketname
    source_file_name = audio_file_path
    audio_file_name = audio_file_path.split(".wav")[-2].split("/")[-1]
    print "audio name: %s" % audio_file_name
    destination_blob_name = audio_file_name

    frame_rate, channels = frame_rate_channel(audio_file_path)
    if channels > 1:
        stereo_to_mono(audio_file_path)


    print "attempting to upload %s to %s as %s" % (source_file_name, bucket_name, destination_blob_name)
    upload_blob(bucket_name, source_file_name, destination_blob_name)


    gcs_uri = 'gs://' + bucketname + '/' + audio_file_name
    transcript = ''

    print "got back gcs uri %s" % gcs_uri

    client = speech.SpeechClient()
    audio = types.RecognitionAudio(uri=gcs_uri)

    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=frame_rate,
        language_code='en-US',
        enable_word_time_offsets=True)

    # Detects speech in the audio file
    operation = client.long_running_recognize(config, audio)
    print "streaming results..."
    response = operation.result(timeout=10000)

    for result in response.results:
        result.alternatives[0].transcript
        transcript += result.alternatives[0].transcript

    # delete_blob(bucket_name, destination_blob_name)
    print "got transcript: "
    print transcript
    return transcript


def write_transcript(transcript, transcript_path):
    with open(output_transcript_path, 'w') as f:
        json.dump(output_transcript_path, f, indent=4)
    f.close()

def stereo_to_mono(audio_file_path):
    sound = AudioSegment.from_wav(audio_file_path)
    sound = sound.set_channels(1)
    sound.export(audio_file_path, format="wav")

def frame_rate_channel(audio_file_path):
    wav_file = wave.open(audio_file_path, "rb")
    frame_rate = wav_file.getframerate()
    channels = wav_file.getnchannels()
    return frame_rate,channels

def upload_transcribe(video_path, transcript_path):
    all_video_files = os.listdir(vid_base_path)
    print "seeing all files:"
    print all_video_files

    for video_file in all_video_files:
        print "now on file " + video_file
        print
        vid_name = video_file.split(".mp4")[0]
        input_vid_path = vid_base_path + '/' + video_file
        output_audio_path = transcript_path + '/' + vid_name + '.wav'

        #/Users/carolynsaund/github/gest-data/data/rock/transcripts/1._History_of_Rock_-_The_Music_Business_in_the_First_Half_of_the_20th_Century-fcva0f6xkDY.wav
        if not os.path.exists(output_audio_path):
            print "creating wav file"
            command = ("ffmpeg -i %s -ab 160k -ac 2 -ar 48000 -vn %s" % (input_vid_path, output_audio_path))
            subprocess.call(command, shell=True)

        transcript = google_transcribe(output_audio_path)
        output_transcript_path = transcript_path + '/' + vid_name + '.json'
        write_transcript(transcript, output_transcript_path)

    ## TODO upload all of these to "audio_bucket_rock_1" gcs buckets

def create_video_subdir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        print ("Creation of the directory %s failed" % dir_path)
    else:
        print ("Successfully created the directory %s " % dir_path)


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

def get_video_transcript(video_path, transcript_path):
    create_video_subdir(transcript_path)
    upload_transcribe(video_path, transcript_path)
    # transcribe_videos(video_path, transcript_path)

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
    print "getting video transcripts"
    get_video_transcript(vid_base_path, transcript_path)
