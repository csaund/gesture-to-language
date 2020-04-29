from subprocess import call
import re

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

"(Ro)(NuSu)(NuSpTx)(SaElTx)(NuSu)(NuSpTx)(SaEl)(SaAtTx)(NuSp)(NuSpTx)(SaElTx)"
"1 thing"
"2 thing"
"3 thing"
"4 thing"
"5 thing"
"6 thing"


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
        return re.sub(r"\s'", "'", text)
    return ""


def get_sequence_encoding(rhet_file):
    content = get_parse_data(rhet_file)
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
    while i < 50:   #len(texts)
        if texts[i] == "":
            i += 1
            continue
        while word_counter < target_word_count:
            print(words[word_counter])
            print(texts[i])
            if texts[i] == "":
                i += 1
                continue
            if words[word_counter] in texts[i]:
                word_counter += 1
                text_indexes.append(i)
                if word_counter == target_word_count - 1:
                    return sorted(list(set(text_indexes)))
                print(words[word_counter])
                print(text_indexes)
            elif i < len(texts)-1 and (texts[i+1] == "" or words[word_counter] in texts[i+1]):
                print("next is blank")
                i += 1
            else:
                print("NO DICE")
                print(texts[i])
                print(words[word_counter])
                word_counter = 0
                text_indexes = []
                i += 1
                break
        if word_counter == target_word_count - 1:
            break
    return list(set(text_indexes))



class RhetoricalAnalyzer:
    def __init__(self):
        return

    def analyze_sentences(self, s_list):
        scores = []
        for s in s_list:
            scores.append(self.analyzer.polarity_scores(s))
        return scores



# want this: (NuLeSpTx)(SaLeElTx)(NuSpSu)(NuLeSpTx)(SaSpEl)(SaLeAtTx)(NuSp)(NuLeSpTx)