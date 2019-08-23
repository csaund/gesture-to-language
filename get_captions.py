from youtube_transcript_api import YouTubeTranscriptApi
import json
import sys


if __name__ == "__main__":
	raw_json = YouTubeTranscriptApi.get_transcript(sys.argv[1], languages=['en'])
 	fn = sys.argv[1] + '.json'	
	with open(fn, 'w') as f:
		json.dump(raw_json, f, indent=4)
	
