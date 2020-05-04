

print("loading modules for Video Manager")
import argparse
from subprocess import call
import cv2
import numpy as np
import os
import pandas as pd
from moviepy.video.io.VideoFileClip import VideoFileClip
import parselmouth


class VideoManager():
    def __init__(self):
        print("initializing Video Manager")
        homePath = os.getenv("HOME")
        self.base_path = os.path.join(homePath, "GestureData")
        self.temp_output_path = os.path.join(self.base_path, 'tmp', 'temp_video.mp4')
        self.df = pd.read_csv(os.path.join(self.base_path, "intervals_df.csv"))

    def get_video_clip(self, video_fn, start_seconds, end_seconds):
        output_path = os.path.join(self.base_path, "videos_clips", video_fn.replace(".mkv", ".mp4").replace(".webm", ".mp4"))
        self.download_video(video_fn, output_path)
        video_fn = video_fn.replace(".mkv", ".mp4").replace(".webm", ".mp4")
        clip_output = video_fn.rsplit('.', 1)[0] + "_" + str(start_seconds) + "_" + str(end_seconds)
        target = clip_output + '.' + video_fn.split('.')[-1]
        print("extracting clip to target: %s" % target)
        with VideoFileClip(output_path) as video:
            new = video.subclip(start_seconds, end_seconds)
            new.write_videofile(target, audio_codec='aac')
        return target

    def get_audio_features(self, video_fn, start_seconds, end_seconds):
        target = self.get_video_clip(self, video_fn, start_seconds, end_seconds)
        #print("target: ", target)
        command = "ffmpeg -i %s -ab 160k -ac 2 -ar 44100 -vn temp_audio.wav" % target
        #print("command: ", command)
        subprocess.call(command, shell=True)
        snd = parselmouth.Sound("temp_audio.wav")
        os.remove("temp_audio.wav")
        return snd

    def download_video(self, video_fn, output_path):
        if os.path.exists(output_path):
            print("error: video already found at path %s" % output_path)
            return
        if not (os.path.exists(os.path.dirname(output_path))):
            os.makedirs(os.path.dirname(output_path))
        err = 0
        print("downloading %s" % video_fn)
        row = self.df.loc[self.df['video_fn'] == video_fn].iloc[0]
        link = row['video_link']
        if 'youtube' not in link:
            print("no youtube video found for video_fn: %s" % link)
            return
        try:
            command = 'youtube-dl -o {path} -f mp4 {link}'.format(link=link, path=output_path)
            res1 = call(command, shell=True)
            print(res1)                       # I think????
            cam = cv2.VideoCapture(output_path)
            # if np.isclose(cam.get(cv2.CAP_PROP_FPS), 29.97, atol=0.03):
            #     os.rename(self.temp_output_path, output_path)
            # else:
            res2 = call('ffmpeg -i "%s" -r 30000/1001 -strict -2 "%s" -y' % (output_path, output_path),
                                shell=True)
            print(("Successfully downloaded: %s" % video_fn))
        except Exception as e:
            print(e)
            err += 1
        finally:
            if os.path.exists(self.temp_output_path):
                print(self.temp_output_path)
                #fo = open(self.temp_output_path, 'r')  # need to close the file so windows can rename it?
                #fo.close()
                #os.remove(self.temp_output_path)
                return