from __future__ import division
#!/usr/bin/env pythons
import json
import os
from google.cloud import storage
devKey = str(open("/Users/carolynsaund/devKey", "r").read()).strip()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/carolynsaund/google-creds.json"

########################################################
############### Getting Actual Keyframes ###############
########################################################
# takes time in form of "X_X_IND_m_s.txt"
def timestring_to_int(time):
    times = time.split("_")
    hrs = int(times[-3])
    mins = int(times[-2])
    sec_and_txt_arr = times[-1].split(".")
    secs = float(sec_and_txt_arr[0])
    if(len(sec_and_txt_arr) == 3):
        secs = secs + float(str("." + str(sec_and_txt_arr[1])))
    timesec = (hrs * 60 * 60) + (mins * 60) + secs
    return timesec

# takes start time in form of X_X_IND_m_s.txt, end time in form of X_X_IND_m_s.txt
# and question time in form of X_X_IND_m_s.txt
# returns whether or not question time is between start and end times
def is_within_time(start_time, end_time, question_time):
    s = timestring_to_int(start_time)
    e = timestring_to_int(end_time)
    q = timestring_to_int(question_time)
    if (q < s) or (q > e):
        return False
    return True

def extract_txt_data(basepath, filepath):
    fn = filepath.split('.txt')[0]
    inf = basepath + filepath
    # print filepath
    with open(inf, 'r') as in_file:
        lines = in_file.read().splitlines()
        x = lines[1].split()
        y = lines[2].split()
        # print x
        # print y
        x = [int(n) for n in x]
        y = [int(n) for n in y]
    return {'x': x, 'y': y}


def get_data_from_path(data_path):
    with open(data_path) as f:
        data = json.load(f)
    return data

def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    print('Blob {} downloaded to {}.'.format(
        source_blob_name,
        destination_file_name))

def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)

def upload_to_gcloud_from_path(fn, path):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(fn)
    blob.upload_from_filename(path)

def read_data(fp):
    with open(fp, 'r') as f:
        t = json.load(f)
    return t

def touch(path):
    with open(path, 'a'):
        os.utime(path, None)

def write_data(fp, data):
    with open(fp, 'w') as f:
        json.dump(data, f, indent=4)
    f.close()