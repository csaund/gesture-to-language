#!/usr/bin/env python
import os
import argparse
import io
import subprocess
import nltk
import json


stop_words = list(set(nltk.corpus.stopwords.words('english')))

## def need to keep the sentence with the id.
def get_sentences(dir_path):
    sentences = []

    for filename in os.listdir(dir_path):
        if filename.endswith(".json"):
            path = dir_path + filename
            with open(path) as f:
                transcript = json.load(f)

                # oh god please no not like this
                for key in transcript.keys():
                    sentences.append(str(transcript[key]))
                    break
        else:
            continue

    return sentences

def process_sentences(sentences):
    for s in sentences:
        tokens = nltk.word_tokenize(s)
        stopped = [w for w in tokens if not w in stop_words]
        # tagged = nltk.pos_tag(tokens)
        # entities = nltk.chunk.ne_chunk(tagged)
        print tokens
        print

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        'path', help='Long mp4 file to be segmented into gestures')
    args = parser.parse_args()

    transcripts_path = './' + args.path + '/'

    sentences = get_sentences(transcripts_path)
    process_sentences(sentences)
