## this is written to run on an ubuntu VM
from __future__ import division
#!/usr/bin/env pythons
import json
import os
# from Discourse_Parser.py import do_parse
# from Discourse_Segmenter.py import do_segment

devKeyPath = os.getenv("devKey")
devKey = str(open(devKeyPath, "r").read()).strip()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = os.path.join(os.getenv("HOME"), "google-creds.json")

from google.cloud import storage
TRANSCRIPT_BUCKET = "audio_transcript_buckets_1"
PARSED_BUCKET = "parsed_transcript_bucket"
POS_BUCKET = "pos_transcript_bucket"
TEMP_TEXT_FILE = "raw_text_tmp.txt"

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
    print('Blob {} downloaded to {}.'.format(
        source_blob_name,
        destination_file_name))
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
    full_transcript = []
    for section in raw_text:
        full_transcript.append((section["transcript"] + ". "))
    outfile = open(TEMP_TEXT_FILE, "a")
    outfile.writelines(full_transcript)
    outfile.close()
    return TEMP_TEXT_FILE


def segment_and_parse(infile, rhet_outfile):
    # run text through segmenter
    # run text through parser

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
    return


def write_and_parse(s, rhet_outfile):
    write_to_file(TEMP_TEXT_FILE, s)
    segment_and_parse(TEMP_TEXT_FILE, rhet_outfile)


def split_segment_parse(fname, fn):
    rhet_outfile = fname + ".rhet_parse"
    f = open(fn, "r")
    raw_text = f.readlines()[0]         # need to cut this up into smaller chunks the parser can handle.
    sentences = raw_text.split(".")     # split by every sentence
    # now make them as large as possible to parse
    s = sentences[0]
    line_count = 0
    for i in range(len(sentences)):
        if i == len(sentences) - 1:
            write_and_parse(s, rhet_outfile)
        elif len(s + sentences[i+1]) < 400:
            s = s + sentences[i+1]
        elif len(s) > 750:          # can normally handle this amount, I think
            s1, s2 = s[:len(s)/2], s[len(s)/2:]
            write_and_parse(s1, rhet_outfile)
            write_and_parse(s2, rhet_outfile)
        else:
            line_count += 1
            write_and_parse(s, rhet_outfile)
            s = sentences[i + 1]
    print(line_count)
    return (rhet_outfile)

    # go through and for each sentence, if it's longer than 400, chop that up as well.
    # can we do that intelligently?

if __name__=="__main__":
    file_list = list_blobs(TRANSCRIPT_BUCKET)
    f = file_list[0]
    print(f)
    temp_json_file = download_blob(TRANSCRIPT_BUCKET, f, "temp.json")
    temp_txt_file = preprocess_json(temp_json_file)
    (rhet_outfile) = split_segment_parse(f, temp_txt_file)
    upload_blob(PARSED_BUCKET, rhet_outfile, rhet_outfile)


    #for f in file_list:
    #    temp_json_file = download_blob(TRANSCRIPT_BUCKET, f, "temp.json")
    #    temp_txt_file = preprocess_json(temp_json_file)
    #    do_segment
    #    do_parse(temp_txt_file)
    #    upload_blob(PARSED_BUCKET, "tmp_doc.dis", f + ".rhet_parse")
    #    upload_blob(POS_BUCKET, "tmp.chp", f + ".pos_tags")


