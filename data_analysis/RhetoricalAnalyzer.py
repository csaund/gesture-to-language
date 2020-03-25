from subprocess import call

## Takes the rhetorical parse located in gs://parsed_transcript_bucket
## Determines similarity of two rhetorical structures of a gesture


class RhetoricalAnalyzer:
    def __init__(self):
        self.analyzer =

    # takes a string that represents a single sentence,
    # returns
    def segment_sentence(self, s):
        command = 'Discourse_Segmenter.py {infile}'.format(infile=s, temp_path=self.temp_output_path)
        res1 = call(command, shell=True)
        python Discourse_Segmenter.py < infile >
        score = self.analyzer.polarity_scores(s)
        return score

    def analyze_sentences(self, s_list):
        scores = []
        for s in s_list:
            scores.append(self.analyzer.polarity_scores(s))
        return scores