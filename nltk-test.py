from sets import Set
import nltk
from nltk.corpus import wordnet as wn

def wnexpand(set):
      res=Set(set)
      #print res
      lst = []
      for w in set:
       for ss in wn.synsets(morph(w)):
         top = Set(ss.lemma_names())
         res = res.union(top)
         for sim in ss.similar_tos():
             res=res.union(Set(sim.lemma_names()))
      for u in res:
       lst.append(u.encode('ascii','ignore'))
      #print lst
      return lst



def morph(w0):
      u = wn.morphy(str(w0))
      if (u == None):
       #print w0
       return w0
      else:
       w = u.encode('ascii','ignore')
       #print w
       return w

#print wnexpand(('big'))
