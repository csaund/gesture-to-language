import xml.dom.minidom
import urllib2
import sys
from urlparse import urlparse
from datetime import datetime
import time

def sort(adict):
  ''' sort a map by keys '''
  items = adict.items()
  items.sort()
  return [value for key, value in items]

# sanity checks
if len(sys.argv) != 2:
  print 'Usage: python ' + str(sys.argv[0]) + ' URL_TO_YOUTUBE_VIDEO'
  print '  f.e. python ' + str(sys.argv[0]) + ' http://www.youtube.com/watch?v=UxnopxbOdic'
  sys.exit(2)

# parse youtube url
url = urlparse(sys.argv[1])
params = dict([part.split('=') for part in url[4].split('&')])
videoId = params['v']
print(videoId)

# fetch annotations xml
response = urllib2.urlopen('http://www.google.com/reviews/y/read2?video_id=' + str(videoId))
whole_thing = response.read()
print(whole_thing)

# parse xml
dom =  xml.dom.minidom.parseString(whole_thing)
annotations = dom.getElementsByTagName("annotation")
texts = {}

for a in annotations:

  text = a.getElementsByTagName("TEXT")
  region = a.getElementsByTagName("rectRegion")

  if not text:
    continue

  if not region:
    region = a.getElementsByTagName("anchoredRegion")

  if region:
    timestamp = region[0].attributes['t'].value

    if timestamp!='never':

      # create timestamp to properly order annotations (second accuracy)
      t = datetime.strptime(timestamp.split('.')[0], "%H:%M:%S")
      t = t.replace(2011)
      texts[time.mktime(t.timetuple())] =  text[0].firstChild.data.strip()

sortedTexts = sort(texts)

# output annotations ordered by time (as appearing in video)
for t in sortedTexts:
  print t
