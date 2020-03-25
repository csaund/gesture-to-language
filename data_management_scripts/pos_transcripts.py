## this is written to run on an ubuntu VM
from __future__ import division
#!/usr/bin/env pythons
import json
import os
from common_helpers import list_blobs, download_blob, upload_blob
import nltk
# import spacy / nltk pos things

devKeyPath = os.getenv("devKey")
devKey = str(open(devKeyPath, "r").read()).strip()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(os.getenv("HOME"), "google-creds.json")

TRANSCRIPT_BUCKET = "audio_transcript_buckets_1"
POS_BUCKET = "pos_transcript_bucket"
TEMP_TEXT_FILE = "raw_text.tmp.txt"
TEMP_JSON_FILE = "raw_json.tmp.json"

def read_data(fn):
    with open(fn, 'r') as f:
        t = json.load(f)
    return t


def write_to_file(fn, text):
    f = open(fn, "w")
    f.writelines(text)
    return


def get_pos_tags(fname, fn):
    pos_outfile = fname + ".pos_tags"
    f = open(fn, "r")
    raw_text = f.readlines()
    pos_tagged = []
    for line in raw_text:
        t = nltk.word_tokenize(line)
        tagged = nltk.pos_tag(t)
        pos_tagged.append(tagged)
    with open(pos_outfile, 'w') as out:
        for item in pos_tagged:
            out.write("%s\n" % item)
    return (pos_outfile)


def preprocess_json(fn):
    raw_text = read_data(fn)
    full_transcript = []
    for section in raw_text:
        full_transcript.append((section["transcript"] + "."))
    with open(TEMP_TEXT_FILE, 'w') as f:
        for item in full_transcript:
            f.write("%s\n" % item)
    return TEMP_TEXT_FILE


if __name__=="__main__":
    file_list = list_blobs(TRANSCRIPT_BUCKET)
    for f in file_list[:3]:
        print(f)
        temp_json_file = download_blob(TRANSCRIPT_BUCKET, f, TEMP_JSON_FILE)
        temp_txt_file = preprocess_json(TEMP_JSON_FILE)
        pos_outfile = get_pos_tags(f, temp_txt_file)
        upload_blob(POS_BUCKET, pos_outfile, pos_outfile)
        os.remove(pos_outfile)
    os.remove(TEMP_TEXT_FILE)
    os.remove(TEMP_JSON_FILE)