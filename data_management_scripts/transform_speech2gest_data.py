print "loading modules"
import argparse
from tqdm import tqdm
import os
import pandas as pd
from common_helpers import *

devKey = str(open("%s/devKey" % os.getenv("HOME"), "r").read()).strip()
bucket_name = "speaker_timings"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "%s/google-creds.json" % os.getenv("HOME")

def convert_time_to_seconds(time):
    intervals = time.split(':')
    # [hours, minutes, seconds.ms]
    seconds = (float(intervals[0]) * 3600) + (float(intervals[1]) * 60) + float(intervals[2])
    return seconds

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-base_path', '--base_path', help='base folder path of dataset', required=True)
    parser.add_argument('-speaker', '--speaker', default='optionally, run only on specific speaker', required=False)

    args = parser.parse_args()
    output_path = args.base_path + '/' + args.speaker + '/timings.json'
    output_name = args.speaker + '_timings.json'

    phrases = []
    print "loading intervals"
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
                "gestures": gestures,
                "speaker": args.speaker
            })
            # input_fn = os.path.join(args.base_path, interval['speaker'], "videos", interval["video_fn"])
            # output_fn = os.path.join(args.output_path, "%s_%s_%s-%s.mp4"%(interval["speaker"], interval["video_fn"], str(start_time), str(end_time)))
            # print(input_fn, output_fn)
            # save_interval(input_fn, str(start_time), str(end_time), output_fn)
        except Exception as e:
            print(e)
            print("couldn't save interval: %s"%interval)

    write_data(output_path, {"phrases": phrases})
    upload_blob(bucket_name, output_path, output_name)
