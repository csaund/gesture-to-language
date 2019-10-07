import argparse
from tqdm import tqdm
import subprocess
import os
import pandas as pd
import json


parser = argparse.ArgumentParser()
parser.add_argument('-base_path', '--base_path', help='base folder path of dataset', required=True)
parser.add_argument('-output_path', '--output_path', default='output directory to save cropped intervals', required=True)
parser.add_argument('-speaker', '--speaker', default='optionally, run only on specific speaker', required=False)

args = parser.parse_args()


## read data from gesture data area
## transform into json I suppose

def convert_time_to_seconds(time):
    # might not work with longer ones?
    intervals = time.split(':')
    # [hours, minutes, seconds.ms]
    seconds = (float(intervals[0]) * 3600) + (float(intervals[1]) * 60) + float(intervals[2])
    return seconds


def write_data(fp, data):
    with open(fp, 'w') as f:
        json.dump(data, f, indent=4)
    f.close()


if __name__ == "__main__":
    phrases = []

    df_intervals = pd.read_csv(os.path.join(args.base_path, 'intervals_df.csv'))
    if args.speaker:
        df_intervals = df_intervals[df_intervals["speaker"] == args.speaker]

    for _, interval in tqdm(df_intervals.iterrows(), total=len(df_intervals)):
        try:
            start_time = str(pd.to_datetime(interval['start_time']).time())
            end_time = str(pd.to_datetime(interval['end_time']).time())
            phase = {
                "video_fn": interval['video_fn'],
                "start_seconds": convert_time_to_seconds(start_time),
                "end_seconds": convert_time_to_seconds(end_time),
                "transcript": ""
            }
            gestures = []
            gestures.append({
                "start_seconds": convert_time_to_seconds(start_time),
                "end_seconds": convert_time_to_seconds(end_time),
            })
            phrases.append({
                "id": interval['interval_id'],
                "phase": phase,
                "gestures": gestures
            })
            # input_fn = os.path.join(args.base_path, interval['speaker'], "videos", interval["video_fn"])
            # output_fn = os.path.join(args.output_path, "%s_%s_%s-%s.mp4"%(interval["speaker"], interval["video_fn"], str(start_time), str(end_time)))
            # print(input_fn, output_fn)
            # save_interval(input_fn, str(start_time), str(end_time), output_fn)
        except Exception as e:
            print(e)
            print("couldn't save interval: %s"%interval)



    write_data(args.output_path, {"phrases": phrases})
