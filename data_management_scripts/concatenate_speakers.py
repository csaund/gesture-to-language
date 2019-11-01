#!/usr/bin/env pythons
import json
import os
import argparse
from common_helpers import *

devKey = str(open("/Users/carolynsaund/devKey", "r").read()).strip()
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/Users/carolynsaund/google-creds.json"

from google.cloud import storage




if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-base_path', '--base_path', help='base folder path of dataset', required=True)
    parser.add_argument('-speakers', '--speakers', default='optionally, run only on specific speaker', nargs='+')
    args = parser.parse_args()

    #vid_base_path = args.base_path + '/' + args.speaker + '/videos'
    #transcript_path = args.base_path + '/' + args.speaker + '/transcripts'

    print args.speakers
