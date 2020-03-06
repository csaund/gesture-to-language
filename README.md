$ Gesture from Language / Language to Gesture


Takes data from UC Berkeleys [gesture data set](http://people.eecs.berkeley.edu/~shiry/speech2gesture/) and clusters gestures by high-level feature vectors detected by motion data. 


### Usage
0. Download the gesture data according to instructions [here](https://github.com/amirbar/speech2gesture/blob/master/data/dataset.md).

1. Transform the data into the handy dandy but very babyish format that I like to use. 
```
$ python transform-speech2gest-data.py --base_path /Users/carolynsaund/github/gest-data/data/ --output_path /Users/carolynsaund/github/gest-data/data/rock/timings.json --speaker rock
```
2. Make sure all the youtube videos are downloaded (only do this on a machine you're happy to have loads of video data on.)
```
$ python download-youtube.py --base_path /Users/carolynsaund/github/gest-data/data/ --speaker rock
```
3. Get the transcripts for all of the youtube videos with timings.
```
$ python get_video_transcripts.py --video_path /Users/carolynsaund/github/gest-data/data/rock/videos --transcript_path /Users/carolynsaund/github/gest-data/data/rock/transcripts [optional] --upload_audio
```
4. Match the transcript data to the gesture data
```
$ python match_transcript_gesture_timings.py --base_path /Users/carolynsaund/github/gest-data/data/ --speaker rock
```
5. Get the gesture clusterer and sentence clusterer going on in your python console
```
$ from GestureSentenceManager import *
$ GSM = GestureSentenceManager("conglomerate_under_10")
$ GSM.downsample_speaker()
$ GSM.cluster_gestures()     ## or GSM.cluster_gestures_under_n_words(10)
$ GSM._initialize_sentence_clusterer()
$ GSM.cluster_sentences_gesture_independent()  # takes a long time. don't try on wimpy machine
$ GSM.test_k_means_gesture_clusters()  ## optional
$ report = GSM.report_clusters()
$ GSM.SentenceClusterer.print_sentences_by_cluster(0)   # if this doesn't work get from 'SentenceClusterer.keys'
$ GSM.assign_gesture_cluster_ids_for_sentence_clusters()
$ GSM.combine_all_gesture_data()
$ An = Analyzer(GSM)          
```
Now you can look at things going on in `GSM.GestureClusterer.clusters`. Mainly the analyzer can show pretty graphs with the methods:
- `Analyzer.plot_semantics_for_gesture(GESTURE_ID)`
- `Analyzer.get_avg_to_mapped_ratio()`

#### Other useful tricks
- Look at similar gesture videos by using 
```
$ a = GSM.GestureClusterer.get_random_gesture_id_from_cluster(4)
$ b = GSM.GestureClusterer.get_random_gesture_id_from_cluster(4)
$ c = GSM.GestureClusterer.get_random_gesture_id_from_cluster(4)
$ GSM.get_gesture_video_clip_by_gesture_id(a)
$ GSM.get_gesture_video_clip_by_gesture_id(b)
$ GSM.get_gesture_video_clip_by_gesture_id(c)
```
this will download actual videos of gestures from the same cluster. Theoretically, these should look pretty good. 




### TODO: 
- compare gesture/sentence cluster overlaps given matchings
- modularize clustering (for sentence and motion data)
- Text clustering 
    - include rhetorical information
    - include syntactic information
    - include sentiment information
- Motion clustering
    - implement more high-level features
    - implement check for normalizing feature difference
    - correlate features/check meaningful features
- Parsing of original data
    - Read text from time around gesture (instead of word number)
    - detect our own gesture phrases  

## Done
* :white_check_mark: Load in dataset

* :white_check_mark: Get video transcript

* :white_check_mark: Match transcript to gesture timings

* :white_check_mark: Cluster those gestures

* :white_check_mark: Cluster sentences organically

* :white_check_mark: Get sentence clusters for gesture clusters

* :white_check_mark: Evaluate in literally any way


## Evaluation
* :white_check_mark: Computationally, see overlap between sentence categories and gesture categories
* :x: Subjectively, see people's matching abilities (gesture --> candidate sentences from categories)

The objective evaluation is bad. 
* silhouette scores avg around 0.5
* clusters largely made up of individuals 
* in terms of movement similarity, matching an incoming sentence to a gesture essentially only gets a floor on how bad the gesture will be, but is no better than average.

### Known Issues:
- Some videos don't get read in correctly, so are missing transcripts. So far those are dropped cause there's plenty more where that came from. 


### TODOs:
- figure out how to trim gestures down to meaningful frames.
- write more/debug high-level features
- Explore clustering of wrist/body data, then re-cluster based on hand motions.
- Visualizations -- chord diagram? network diagram? 
- Include 
- new sentence clusterings using nltk: Wu-Palmer, Jiang-Conrath distance.
- metric to compare quality of clustering 
- include prosody/"beat" section?
- create separate clusters for each speech feature!!!



#### Separating Clusters
An individual gesture does not necessarily emphasize EVERYTHING about a sentence. Instead, 
a gesture emphasizes *part* of what an individual is trying to say (i.e. the metaphoric, affective, relationship between subjects, etc).
Thus, one general text clustering doesn't necessarily make sense, as gestures are not a blend of 
all text features, but generally an emphasis of one. 

So, it may make more sense to create, say, 5 separate text clusterings and THEN do a mapping to the 
motion clusters, and evaluate THOSE mappings. 

As of now, only semantic text clustering is really implemented. We're really only expecting to
see any sort of motion cluster with certain adjectives or adverbs. It makes sense that with the shitty clusters
we're getting shitty results.  