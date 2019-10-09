#!/usr/bin/env pythons
print "importing libs"
import argparse
import pandas as pd



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('-base_path', '--base_path', help='base folder path of dataset', required=True)
    #parser.add_argument('-output_path', '--output_path', default='output directory to save wav files', required=True)
    parser.add_argument('-speaker', '--speaker', default='optionally, run only on specific speaker', required=False)
    parser.add_argument('-frames', '--frames', default='path to frames file', required=False)

    args = parser.parse_args()

    print "reading csv from " + args.frames
    ## 7.8M frames
    frames = pd.read_csv(args.frames)

    frames.shape
