from __future__ import unicode_literals

import argparse
from subprocess import call

import cv2
import numpy as np
import os
import pandas as pd
from tqdm import tqdm
import youtube_dl

parser = argparse.ArgumentParser()
parser.add_argument('-base_path', '--base_path', help='base folder path of dataset')
parser.add_argument('-speaker', '--speaker',
                    help='download videos of a specific speaker {oliver, jon, conan, rock, chemistry, ellen, almaram, angelica, seth, shelly}')
args = parser.parse_args()

BASE_PATH = args.base_path
df = pd.read_csv(os.path.join(BASE_PATH, "intervals_df.csv"))

if args.speaker:
    df = df[df['speaker'] == args.speaker]

err = 0
temp_output_path = '/tmp/temp_video.mp4'
commands = []

for _, row in tqdm(df.iterrows(), total=df.shape[0]):
    # print(row)
    # i, name, link, ds, s, e = row
    link = row["video_link"]
    if 'youtube' in link:
        try:
            output_path = os.path.join(BASE_PATH, row["speaker"], "videos", row["video_fn"])
            if not (os.path.exists(os.path.dirname(output_path))):
                os.makedirs(os.path.dirname(output_path))
            command = 'youtube-dl -o {temp_path} -f mp4 {link}'.format(link=link, temp_path=temp_output_path)
            # super hacky way to not download things more than once.
            if command not in commands:
                res1 = call(command, shell=True)
                commands.append(command)
                cam = cv2.VideoCapture(temp_output_path)
                if np.isclose(cam.get(cv2.CAP_PROP_FPS), 29.97, atol=0.03):
                    os.rename(temp_output_path, output_path)
                else:
                    res2 = call('ffmpeg -i "%s" -r 30000/1001 -strict -2 "%s" -y' % (temp_output_path, output_path),
                                shell=True)
        except Exception as e:
            print e
            err += 1
        finally:
            if os.path.exists(temp_output_path):
                os.remove(temp_output_path)
print("Successfully downloaded: %s/%s" % (len(df) - err, len(df)))

## from https://github.com/amirbar/speech2gesture/blob/master/data/download/download_youtube.py
