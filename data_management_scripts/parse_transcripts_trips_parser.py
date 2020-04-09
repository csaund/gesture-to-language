#!/usr/bin/env pythons
import os
import requests
from common_helpers import *
# import xml.etree.ElementTree as ET        # will do this in the rhetorical clusterer.
from tqdm import tqdm
import time

devKeyPath = os.getenv("devKey")
devKey = str(open(devKeyPath, "r").read()).strip()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(os.getenv("HOME"), "Downloads", "google-creds.json")

from google.cloud import storage
TRANSCRIPT_BUCKET = "audio_transcript_buckets_1"
PARSED_BUCKET = "trips_parsed_transcript_bucket"
PARSER_ENDPOINT = "http://trips.ihmc.us/parser/cgi/parse"
PARAMS = {'input': ""}
TEMP_JSON_FILE = "temp.json"
TEMP_TEXT_FILE = "raw_text_full_tmp.txt"
TEMP_PARTIAL_TEXT_FILE = "raw_text_partial_tmp.txt"
SEGMENTED_FILE = "segmented.txt"

def write_text(txt, fn):
    with open(fn, 'w') as f:
        f.writelines(txt)


# take the json from json into a full transcript that can be parsed by the rhetorical parser.
def preprocess_json(fn):
    raw_text = read_data(fn)
    full_transcript = []
    for section in raw_text:
        full_transcript.append((section["transcript"] + "."))
    return " ".join(full_transcript)


# actually send to the parser so we can append the xml.
def send_to_parser(txt, outfile):
    PARAMS['input'] = txt
    print("got params")
    print(PARAMS)
    r = requests.post(url=PARSER_ENDPOINT, data=PARAMS)
    if r.ok:
        # print(r.encoding)     # do we need to always make sure utf-8? should always be fine.
        data = r.text
        rhet_out = open(outfile, "a+")
        rhet_out.writelines(data)
    else:
        print("ERR, got non-ok response code %s" % r.status_code)
        print(r.reason)
    return


# split text into chunks smaller than n characters while preserving words.
def split_text(txt, n=2000):
    chunks = []
    sentence = ""
    words = txt.split(" ")
    for w in words:
        if (len(sentence) + len(w) + 2) < n:    # make the longest thing you can
            sentence += w + " "
        else:       # when you can't make anymore, upload it.
            chunks.append(sentence)
            sentence = w
    # at the end get the leftovers
    chunks.append(sentence)
    return chunks


if __name__ == "__main__":
    file_list = list_blobs(TRANSCRIPT_BUCKET)
    already_parsed = list_blobs(PARSED_BUCKET)
    for f in file_list[:1]:
        if f+'.parse.xml' in already_parsed:
            continue
        rhet_outfile = f + '.parse.xml'
        temp_json_file = download_blob(TRANSCRIPT_BUCKET, f, TEMP_JSON_FILE)
        raw_text = preprocess_json(TEMP_JSON_FILE)
        splits = split_text(raw_text)   # input must be shorter than 2000 chars to send to the parser
        for s in tqdm(splits):
            print(s)
            send_to_parser(s, rhet_outfile)
            time.sleep(15)

        upload_blob(PARSED_BUCKET, rhet_outfile, rhet_outfile)

        os.remove(rhet_outfile)
        os.remove(TEMP_JSON_FILE)
