#!/usr/bin/env python
print "importing libs"
import os
import io
import argparse
import subprocess
import json
## sound stuff
import wave
from pydub import AudioSegment
## Google stuff
from google.cloud import storage
from google.cloud import speech
from google.cloud.speech import enums
from google.cloud.speech import types
## ffmpeg stuff
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from oauth2client.client import GoogleCredentials
from tqdm import tqdm

devKey = str(open("/Users/carolynsaund/devKey", "r").read()).strip()
bucketname = "audio_bucket_rock_1"
transcript_bucketname = "audio_transcript_buckets_1"

from apiclient.discovery import build
service = build('language', 'v1', developerKey=devKey)
collection = service.documents()

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/carolynsaund/google-creds.json"

# very very helpful google tutorial
# https://towardsdatascience.com/how-to-use-google-speech-to-text-api-to-transcribe-long-audio-files-1c886f4eb3e9


def get_audio_filename_from_path(fp):
    return fp.split(".wav")[-2].split("/")[-1]


def get_transcript_filepath_from_audio_path(fp):
    return fp.replace(".wav", ".json")


def get_transcript_from_transcript_filepath(fp):
    print "getting transcript from file %s" % fp
    with open(fp, 'r') as f:
        t = json.load(f)
        return t

def get_audio_filepath_from_video_path(fp):
    return fp.replace("videos", "transcripts").replace("mp4", "wav")


def write_transcript(transcript, transcript_path):
    with open(transcript_path, 'w') as f:
        json.dump(transcript, f, indent=4)
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


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)


def write_transcript(transcript, transcript_path):
    with open(transcript_path, 'w') as f:
        json.dump(transcript, f, indent=4)
    f.close()


def upload_transcript(transcript_name, transcript_path):
    print ("uploading %s from %s to %s" % (transcript_name, transcript_path, transcript_bucketname))
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(transcript_bucketname)
    blob = bucket.blob(transcript_name)
    blob.upload_from_filename(transcript_path)


def upload_audio(audio_file_path):
    bucketname = "audio_bucket_rock_1"
    bucket_name = bucketname
    print "uploading %s to %s" % (audio_file_path, bucket_name)
    audio_file_name = get_audio_filename_from_path(audio_file_path)
    destination_blob_name = audio_file_name

    frame_rate, channels = frame_rate_channel(audio_file_path)
    if channels > 1:
        stereo_to_mono(audio_file_path)

    # upload so we can get a gcs for ourselves.
    upload_blob(bucket_name, audio_file_path, destination_blob_name)
    gcs_uri = 'gs://' + bucketname + '/' + audio_file_name


#### NEW ####
def google_transcribe(audio_file_path):
    found_previous_transcript = False
    print "attempting to transcribe file %s" % audio_file_path
    bucket_name = bucketname
    audio_file_name = get_audio_filename_from_path(audio_file_path)

    # ## if we've already transcribed this video, we're done here.
    if os.path.exists(get_transcript_filepath_from_audio_path(audio_file_path)):
        print "already found transcript for that audio path."
        found_previous_transcript = True
        return (get_transcript_from_transcript_filepath(get_transcript_filepath_from_audio_path(audio_file_path)), found_previous_transcript)

    gcs_uri = 'gs://' + bucketname + '/' + audio_file_name

    frame_rate, channels = frame_rate_channel(audio_file_path)
    client = speech.SpeechClient()
    audio = types.RecognitionAudio(uri=gcs_uri)
    config = types.RecognitionConfig(
        encoding=enums.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=frame_rate,
        language_code='en-US',
        enable_word_time_offsets=True)

    # Detects speech in the audio file
    operation = client.long_running_recognize(config, audio)
    print "getting results..."
    response = operation.result(timeout=10000)

    transcript = []
    for result in response.results:
        out = {}
        alternative = result.alternatives[0]
        out["transcript"] = str(alternative.transcript)
        out["words"] = []
        for i in range(len(alternative.words)):
            word_info = alternative.words[i]
            ## nanos is super f'ed up in these responses? like they go up and down a bunch?
            ## let's just try with seconds and see where we get.
            word_start = word_info.start_time.seconds
            word_end = word_info.end_time.seconds
            w = word_info.word
            word = {
                "word": w,
                "word_start": word_start,
                "word_end": word_end
            }
            out['words'].append(word)
        transcript.append(out)

    ## TODO uncomment/implement if I want to do this.
    # this resource is brilliant:
    # https://towardsdatascience.com/how-to-use-google-speech-to-text-api-to-transcribe-long-audio-files-1c886f4eb3e9
    # delete_blob(bucket_name, destination_blob_name)
    print "got transcript"
    return (transcript, found_previous_transcript)


def get_audio_from_video(vid_name, id_base_path, transcript_path):
    input_vid_path = vid_base_path + '/' + vid_name + '.mp4'
    output_audio_path = transcript_path + '/' + vid_name + '.wav'
    if(os.path.exists(output_audio_path)):
        return
    print "creating wav file"
    command = ("ffmpeg -i %s -ab 160k -ac 2 -ar 48000 -vn %s" % (input_vid_path, output_audio_path))
    subprocess.call(command, shell=True)
    return


def process_video_files(vid_base_path, transcript_base_path):
    print "Getting speech results from videos"
    all_video_files = os.listdir(vid_base_path)
    for video_file in tqdm(all_video_files):
        ## TODO simplify this
        vid_name = video_file.split(".mp4")[0]
        output_audio_path = transcript_base_path + '/' + vid_name + '.wav'
        transcript_path = transcript_base_path + '/' + vid_name + '.json'

        transcript_name = vid_name + '.json'

        #  get_audio_from_video(vid_name, vid_base_path, transcript_base_path)
        (transcript, found_previous_transcript) = google_transcribe(output_audio_path)
        # will not rewrite previous file
        if not found_previous_transcript:
            print "No previous file found for %s" % transcript_path
            write_transcript(transcript, transcript_path)
            upload_transcript(transcript_name, transcript_path)
            continue
        print "Previous file found for %s. Not overwriting." % transcript_path
        upload_transcript(transcript_name, transcript_path)

def get_wavs_from_video(vid_base_path, transcript_base_path):
    print "Generating wavs from video"
    all_video_files = os.listdir(vid_base_path)
    for video_file in tqdm(all_video_files):
        vid_name = video_file.split(".mp4")[0]
        output_audio_path = transcript_base_path + '/' + vid_name + '.wav'
        if ".mkv." in output_audio_path:
            continue
        get_audio_from_video(vid_name, vid_base_path, transcript_base_path)
        ## TODO make this whole thing respond well to multiple video types
        upload_audio(output_audio_path)


def create_video_subdir(dir_path):
    try:
        os.mkdir(dir_path)
    except OSError:
        print ("Creation of the directory %s failed" % dir_path)
    else:
        print ("Successfully created the directory %s " % dir_path)


def get_video_transcripts(video_path, transcript_path):
    create_video_subdir(transcript_path)
    get_wavs_from_video(video_path, transcript_path)
    process_video_files(video_path, transcript_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-base_path', '--base_path', help='base folder path of dataset', required=True)
    parser.add_argument('-speaker', '--speaker', default='optionally, run only on specific speaker', required=False)
    args = parser.parse_args()

    vid_base_path = args.base_path + '/' + args.speaker + '/videos'
    transcript_path = args.base_path + '/' + args.speaker + '/transcripts'

    get_video_transcripts(vid_base_path, transcript_path)