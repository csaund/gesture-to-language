from __future__ import unicode_literals

print "loading modules for Video Manager"
import argparse
from subprocess import call

import cv2
import numpy as np
import os
import pandas as pd
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
ffmpeg_extract_subclip("video1.mp4", start_time, end_time, targetname="test.mp4")


class VideoManager():
    def __init__(self):
        print "initializing Video Manager"
        self.base_path = "%s/github/gest-data/data" % os.getenv("HOME")
        self.temp_output_path = '/tmp/temp_video.mp4'
        self.df = pd.read_csv(os.path.join(self.base_path, "intervals_df.csv"))

    def get_video_clip(self, video_fn, start_seconds, end_seconds):
        output_path = os.path.join(self.base_path, row["speaker"], "videos", row["video_fn"])
        self.download_video(video_fn, output_path)
        clip_output = video_fn.split('.')[0] + "_" + str(start_seconds) + "_" + str(end_seconds)
        target = clip_output + '.' + video_fn.split('.')[-1]
        ffmpeg_extract_subclip(output_path, start_seconds, end_seconds, targetname=target)
        return

    def download_video(self, video_fn, output_path):
        if os.path.exists(output_path):
            print "error: video already found at path %s" % output_path
            return
        if not (os.path.exists(os.path.dirname(output_path))):
            os.makedirs(os.path.dirname(output_path))
        err = 0
        print "downloading %s" % video_fn
        row = df.loc[df['video_fn'] == video_fn]
        link = row['video_link']
        if 'youtube' not in link:
            print "no youtube video found for video_fn: %s" % video_fn
            return
        try:
            command = 'youtube-dl -o {temp_path} -f mp4 {link}'.format(link=link, temp_path=self.temp_output_path)
            res1 = call(command, shell=True)
            commands.append(command)
            cam = cv2.VideoCapture(temp_output_path)
            if np.isclose(cam.get(cv2.CAP_PROP_FPS), 29.97, atol=0.03):
                os.rename(temp_output_path, output_path)
            else:
                res2 = call('ffmpeg -i "%s" -r 30000/1001 -strict -2 "%s" -y' % (self.temp_output_path, output_path),
                                shell=True)
        except Exception as e:
            print e
            err += 1
        finally:
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
        print("Successfully downloaded: %s/%s" % (len(df) - err, len(df)))
