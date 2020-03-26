## this is written to run on an ubuntu VM
from __future__ import division
#!/usr/bin/env pythons
import json
import os
import time
import subprocess
from tqdm import tqdm
# actually have to call these as a subprocess
# from Discourse_Parser import do_parse
# from Discourse_Segmenter import do_segment

devKeyPath = os.getenv("devKey")
devKey = str(open(devKeyPath, "r").read()).strip()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(os.getenv("HOME"), "Downloads", "google-creds.json")

from google.cloud import storage
TRANSCRIPT_BUCKET = "audio_transcript_buckets_1"
PARSED_BUCKET = "parsed_transcript_bucket"
POS_BUCKET = "pos_transcript_bucket"
TEMP_TEXT_FILE = "raw_text_full_tmp.txt"
TEMP_PARTIAL_TEXT_FILE = "raw_text_partial_tmp.txt"
SEGMENTED_FILE = "segmented.txt"

# to be run wherever Discourse Parser is installed (http://alt.qcri.org/tools/discourse-parser/)
# takes all transcripts in gs://audio_transcript_buckets_1/
# sends through discourse parser
# puts result of parsed transcript in gs://parsed_transcript_bucket/

def read_data(fp):
    with open(fp, 'r') as f:
        t = json.load(f)
    return t


def download_blob(bucket_name, source_blob_name, destination_file_name):
    """Downloads a blob from the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob_name)
    blob.download_to_filename(destination_file_name)
    return destination_file_name


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(source_file_name)
    return


def list_blobs(bucket_name=TRANSCRIPT_BUCKET):
    """Lists all the blobs in the bucket."""
    # bucket_name = "your-bucket-name"
    storage_client = storage.Client()
    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(bucket_name)
    f_names = []
    for blob in blobs:
        f_names.append(blob.name)
    return f_names


# take the json from json into a full transcript that can be parsed by the rhetorical parser.
def preprocess_json(fn):
    raw_text = read_data(fn)
    # print(len(raw_text))
    full_transcript = []
    for section in raw_text:
        full_transcript.append((section["transcript"] + "."))
    with open(TEMP_TEXT_FILE, 'w') as f:
        for item in full_transcript:
            f.write("%s\n" % item)
    return TEMP_TEXT_FILE


def segment_and_parse(infile, rhet_outfile):
    # run text through segmenter
    subprocess.call(['python','Discourse_Segmenter.py',infile])
    subprocess.call(['cat', SEGMENTED_FILE])
    # do_segment(infile)   # for some reason this isn't very happy.
    # run text through parser
    subprocess.call(['python', 'Discourse_Parser.py', SEGMENTED_FILE])
    # do_parse(SEGMENTED_FILE)
    # append to appropriate outfiles
    rhet_out = open(rhet_outfile, "a+")
    rhet_in = open("tmp_doc.dis", "r")
    rhet_d = rhet_in.readlines()
    rhet_d.insert(0, "\n")
    rhet_out.writelines(rhet_d)
    return


def write_to_file(fn, text):
    f = open(fn, "w")
    f.writelines(text)
    f.close()
    return 

def write_and_parse(s, rhet_outfile):
    write_to_file(TEMP_PARTIAL_TEXT_FILE, s)
    subprocess.call(['cat', 'raw_text_partial_tmp.txt'])
    # segment_and_parse(TEMP_PARTIAL_TEXT_FILE, rhet_outfile)


def split_segment_parse(fname, fn):
    rhet_outfile = fname + ".rhet_parse"
    f = open(fn, "r")
    sentences = f.readlines()         # need to cut this up into smaller chunks the parser can handle.
    # sentences = raw_text.split(".")     # split by every sentence
    # now make them as large as possible to parse
    s = sentences[0]
    short_sentences = []
    for i in range(len(sentences)):
        if i == len(sentences) - 1:
            short_sentences.append(s)
        elif len(s + sentences[i+1]) < 400:
            s = s + sentences[i+1]
        elif len(s) > 550:          # can normally handle this amount, I think
            # need to split the string such that sentences are preserved 
            # as much as possible. or at least words. 
            words = s.split(" ")
            # print(words)
            w1, w2 = words[:int(len(words)/2)], words[int(len(words)/2):]
            # print("WORDS 1")
            # print(w1)
            # print("WORDS 2")
            # print(w2)
            s1, s2 = " ".join(w1), " ".join(w2)
            short_sentences.append(s1)
            short_sentences.append(s2)
            s = sentences[i + 1]
        else:
            short_sentences.append(s)
            s = sentences[i + 1]
    # now go through and actually parse them
    for sent in tqdm(short_sentences):
        print(sent)
        write_and_parse(sent, rhet_outfile)

    return (rhet_outfile)

    # go through and for each sentence, if it's longer than 400, chop that up as well.
    # can we do that intelligently?

if __name__=="__main__":
    file_list = list_blobs(TRANSCRIPT_BUCKET)
    f = file_list[0]
    # print(f)
    temp_json_file = download_blob(TRANSCRIPT_BUCKET, f, "temp.json")
    temp_txt_file = preprocess_json(temp_json_file)
    (rhet_outfile) = split_segment_parse(f, temp_txt_file)
    # upload_blob(PARSED_BUCKET, rhet_outfile, rhet_outfile)



