
# a generalized Clusterer class that requires its caller
# to have a `compute_difference` function that compares the difference
# in its given domain.
# optionally takes seeds which can be used to start the clusterer.
class Clusterer():

    def __init__(self, caller, seeds, data=None):
        self.caller = caller
        self.seeds = seeds
        self.data = data

    # PLEASE PLEASE PLEASE put in your own k :(
    def cluster(self, data=None, k=15):
        dat = data if data else self.data
        for d in dat:
