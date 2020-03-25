from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# analyzes sentences according to https://github.com/cjhutto/vaderSentiment#python-code-example
# so returns in format:
# {
#   'pos': 0.746,
#   'compound': 0.8316,
#   'neu': 0.254,
#   'neg': 0.0
# }

class SentimentAnalyzer:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()

    # takes a string that represents a single sentence,
    # returns
    def analyze_sentence(self, s):
        score = self.analyzer.polarity_scores(s)
        return score

    def analyze_sentences(self, s_list):
        scores = []
        for s in s_list:
            scores.append(self.analyzer.polarity_scores(s))
        return scores
