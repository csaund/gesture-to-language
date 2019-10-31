# gesture-to-language


Takes data from UC Berkeleys [gesture data set](http://people.eecs.berkeley.edu/~shiry/speech2gesture/) and clusters gestures by high-level feature vectors detected by motion data. 

0. Download the gesture data according to instructions. 

1. Transform the data into the handy dandy but very babyish format that I like to use. 
```
$ python transform-speech2gest-data.py --base_path /Users/carolynsaund/github/gest-data/data/ --output_path /Users/carolynsaund/github/gest-data/data/rock/timings.json --speaker rock
```
2. Make sure all the youtube videos are downloaded (only do this on a machine you're happy to have loads of video data on.)
```
$ sudo python download-youtube.py --base_path /Users/carolynsaund/github/gest-data/data/ --speaker rock
```
3. Get the transcripts for all of the youtube videos with timings.
```
$ sudo python get_video_transcripts.py --video_path /Users/carolynsaund/github/gest-data/data/rock/videos --transcript_path /Users/carolynsaund/github/gest-data/data/rock/transcripts [optional] --upload_audio
```
4. Match the transcript data to the gesture data
```
$ sudo python match_transcript_gesture_timings.py --base_path /Users/carolynsaund/github/gest-data/data/ --speaker rock
```
5. Get the gesture clusterer and sentence clusterer going on.
```
$ GSM = GestureSentenceManager("/Users/carolynsaund/github/gest-data/data", "rock")
$ GSM.load_gestures()
$ GSM.cluster_gestures()    ## about 10 minutes
$ report = GSM.report_clusters()
$ SC = SentenceClusterer("/Users/carolynsaund/github/gest-data/data", "rock")   
$ SC.cluster_sentences([],[optional: min cluster similarity)    ## about 20 minutes
```
Now you can look at things going on.

### TODO: 
- conglomerate gesture/sentence clusterer. 
- compare gesture/sentence cluster overlaps given matchings


## What I want
* :white_check_mark: Load in dataset

* :white_check_mark: Get video transcript

* :white_check_mark: Match transcript to gesture timings

* :white_check_mark: Cluster those gestures

* :white_check_mark: Cluster sentences organically

* :x: Get sentence clusters for gesture clusters

* :x: Evaluate in literally any way


## Evaluation
* Computationally, see overlap between sentence categories and gesture categories
* Subjectively, see people's matching abilities (gesture --> candidate sentences from categories)

### Known Issues:


### TODOs:
- figure out how to trim gestures down to meaningful frames...
- write more/debug high-level features
- Explore clustering of wrist/body data, then re-cluster based on hand motions...
- Visualizations -- chord diagram? network diagram? 
- new sentence clusterings using nltk: Wu-Palmer, Jiang-Conrath distance.
- metric to compare quality of clustering 
