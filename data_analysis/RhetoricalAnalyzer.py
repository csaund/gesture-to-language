from subprocess import call
import re
import os
from common_helpers import download_blob

## Takes the rhetorical parse located in gs://parsed_transcript_bucket
## Determines similarity of two rhetorical structures of a gesture
KEY_TERMS = {
    "Root": "Ro",
    "Nucleus": "Nu",
    "Satellite": "Sa",
    "leaf": "Le",
    "span": "Sp",
    "Elaboration": "El",
    "Attribution": "At",
    "Same-Unit": "Su",
    "Cause": "Ca",
    "Temporal": "Te",
    "Background": "Ba",
    "Enablement": "En",
    "Joint": "Jo",
    "Contrast": "Co",
    "Condition": "Cd",
    "Manner-Means": "Mm",
    "Explanation": "Ex",
    "Comparison": "Cm",
    "text": "Tx"
}


CONST_REPLACEMENTS = {
    " '": "'",
    " n't": "n't"
}

# https://stackoverflow.com/questions/15175142/how-can-i-do-multiple-substitutions-using-regex-in-python
def multi_replace(text, replacement_dict=None):
    if replacement_dict is None:
        replacement_dict = CONST_REPLACEMENTS
    regex = re.compile("(%s)" % "|".join(map(re.escape, replacement_dict.keys())))
    # For each match, look-up corresponding value in dictionary
    return regex.sub(lambda mo: replacement_dict[mo.string[mo.start():mo.end()]], text)


def get_parse_data(fp):
    f = open(fp, "r")
    rhet_lines = f.readlines()
    f.close()
    return rhet_lines


def get_line_encoding(line):
    encoding = "("
    for k in KEY_TERMS.keys():
        if k in line:
            encoding += KEY_TERMS[k]
    encoding += ")"
    return encoding


def get_text_splices(line):
    if "_!" in line:
        text = line.split("_!")[-2]
        return multi_replace(text)
    return ""


def get_sequence_encoding(content=None, rhet_file=None):
    content = content if content else get_parse_data(rhet_file)
    texts = []
    encoding = ""
    for line in content:
        temp = get_line_encoding(line)
        if temp != "()":
            texts.append(get_text_splices(line))
            encoding += temp
    en = ("".join(encoding.split("(")[1:])).split(")")
    return en, texts


# doesn't work perfectly but works well enough for now lol fml.
def get_matching_words(transcript, texts):
    words = transcript.split(" ")
    target_word_count = len(words)
    word_counter = 0
    text_indexes = []
    i = 0
    while i < len(texts):
        if texts[i] == "":
            i += 1
            continue
        while word_counter < target_word_count:
            if texts[i] == "":
                i += 1
                continue
            if words[word_counter] in texts[i]:
                word_counter += 1
                text_indexes.append(i)
                if word_counter == target_word_count - 1:
                    return sorted(list(set(text_indexes)))
            elif i < len(texts)-1 and (texts[i+1] == "" or words[word_counter] in texts[i+1]):
                i += 1
            else:
                word_counter = 0
                text_indexes = []
                i += 1
                break
        if word_counter == target_word_count - 1:
            break
    return sorted(list(set(text_indexes)))


def get_rhetorical_encoding_for_gesture(g):
    transcript_to_match = g['phase']['transcript']
    rhetorical_parse_file = g['phase']['video_fn'][:-3] + "rhet_parse"
    en, texts = get_sequence_encoding(rhetorical_parse_file)
    text_range = get_matching_words(transcript_to_match, texts)
    if not text_range:
        print("No rhetorical encoding found for gesture ", g['id'])
        return None
    return en[text_range[0]:text_range[-1]+1]


class RhetoricalAnalyzer:
    def __init__(self):
        self.bucket = "parsed_transcript_bucket"
        return

    def get_sequence_for_gesture(self, g):
        transcript_to_match = g['phase']['transcript']
        rhetorical_parse_file = g['phase']['video_fn'][:-3] + "json.rhet_parse"
        # download and delete?
        f = download_blob(self.bucket, rhetorical_parse_file, "tmp.rhet")
        content = get_parse_data("tmp.rhet")
        os.remove("tmp.rhet")
        en, texts = get_sequence_encoding(content=content)
        text_range = get_matching_words(transcript_to_match, texts)
        if not text_range:
            print("No rhetorical encoding found for gesture ", g['id'])
            return None
        return en[text_range[0]:text_range[-1] + 1]




