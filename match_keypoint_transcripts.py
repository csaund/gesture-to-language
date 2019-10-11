#!/usr/bin/env python
print "importing libs"
import argparse
import pandas as pd
import itertools
import csv
import math
import json
import glob, os
import io
import subprocess







if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-base_path', '--base_path', help='base folder path of dataset', required=True)
    parser.add_argument('-output_path', '--output_path', default='output directory to save wav files', required=True)
    parser.add_argument('-speaker', '--speaker', default='optionally, run only on specific speaker', required=False)
    # --base_path /Users/carolynsaund/github/gest-data/data --speaker rock
    gesture_keypoints =
