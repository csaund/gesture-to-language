# gesture-to-language


Alright so here we go the first thing's first is you can cut a video using this command:
`ffmpeg -i megyn-kelly.mp4 -ss 00:06:07 -to 00:06:09 -c:v copy -c:a copy test.mp4`

Then you can turn that into a wav that you can get the transcript for like this:
`ffmpeg -i test.mp4 -ab 160k -ac 2 -ar 48000 -vn audio_test.wav`

Then you can get the transcript for that like this:
`sudo python transcribe_word_time_offsets.py audio_test.wav`

And that will put the output into a file called `audio_test.json`

Rad.

From there I need to do nltk things. Time to install wordnet I guess!

usage
```
$ python segment-long-video.py video-file.mp4
```
requires that you first have video-file-timings.json which must be in the format
```

{
  "phrases":
  [
      {
        "id": 1,
        "phase": {
            "start_seconds": 363,
            "end_seconds": 370.7,
            "transcript": ""
        },
        "gestures": [
            {
              "start_seconds": 364,
              "end_seconds": 365.5
            },
            {
              "start_seconds": 365.6,
              "end_seconds": 366
            },
            {
              "start_seconds": 367.7,
              "end_seconds": 369
            },
            {
              "start_seconds": 369,
              "end_seconds": 370.6
            }
        ]
      },
      ...
  ```

creates a folder `video-file/` that contains the videos of gesture phrases and audio files (`.mp4`/`.wav`), and json files with associated transcript and timings for each gesture phrase and sub-gesture, as given by the `-timings.json` file.

so you'd do two things to get all the final forms:
```
$ python segment-long-video.py megyn-kelly.mp4
$ python analyze-transcripts.py megyn-kelly.mp4
```


## :fire: Hot :fire: Update
This is all about using the MIT individual gesture dataset, which I think I can do after having downloaded the dataset and running the following commands:
1. Get the timings into a format I originally made
```
$ python transform-speech2gest-data.py --base_path /Users/carolynsaund/github/gest-data/data/ --output_path /Users/carolynsaund/github/gest-data/data/rock/timings.json --speaker rock
```
2. Make sure all the youtube videos are downloaded (only do this on the cluster)
```
$ sudo python download-youtube.py --base_path /Users/carolynsaund/github/gest-data/data/ --speaker rock
```
3. You can get the frame data (more or less)
```
$ sudo python analyze-frames.py --base_path /Users/carolynsaund/github/gest-data/data --speaker rock
```
4. TODO: actually match the gesture keyframes to the transcript.
    - need to get the transcript for the longer video, but run into limits from google API
    - get around by asking for smaller segments of time? Like getting, perhaps, only 50 seconds of audio at a time and matching it to the gesture timings?



## What I want
* :white_check_mark: A script that takes a long video

* :white_check_mark: Segments that video into gesture phrases and sub-gestures

* :x: Takes those mini-gesture videos, and classifies the gesture by movement characteristics

* :white_check_mark: Takes those mini-gesture videos, and gets the transcript for them

* :white_check_mark: Takes those transcripts, and gets their linguistic forms (nltk?)

* :heavy_minus_sign: Takes those linguistic forms, categorize them into linguistic categories by:

  * :white_check_mark: sentiment

  * :white_check_mark: tfidf

  * :x: syntax

* :x: matches up and compares gestures <--> sentences that accompany them


## Evaluation
* Computationally, see overlap between sentence categories and gesture categories
* Subjectively, see people's matching abilities (gesture --> candidate sentences from categories)

### Known Issues:
- Currently if a gesture goes to the last frame of the file, it doesn't load the last frame due to a bug in `get_keyframes_per_gesture`. Can't be arsed to fix it because I'm pretty sure I know why it happens and it's literally only the last frame that it happens on.
- I am very sketched out by the movements when they go into the last frame. It shows movement/keyframes in time periods when the screen is black, so I need to look more into OpenPose to see how they report keyframes when the skeleton seems to disappear... Seems like the gesture should have been cut off but I'm also not sure about how Berkeley did their gesture segementation.
- Also sketchy is a low number of frames for some gestures.

### TODOs:
- figure out how to trim gestures down to meaningful frames...
- figure out how to compare gestures of different numbers of frames
    - pad shorter gesture with same value for rest of sequence?
    - best most common sub-sequence of some minimal frame length...
    - but needs to be the difference between each point and the last point
    - but also needs to take into account difference between raw positions (ex. flat vs vertical spread palm)
- Explore clustering of wrist/body data, then re-cluster based on hand motions...
- after getting all features, need to normalize across all gestures
    - and after that, everything that is "minimum" need to change to 1-n


YEEEEE DOGGIE let's get transcripts from all our speakers. 
`$ python download_youtube.py --base_path PATH_TO_DATA --speaker SPEAKER_YOU_WANT`
`$ python transform_speech2gest_data.py --base_path PATH_TO_DATA --speaker SPEAKER_YOU_WANT`
`$ python get_video_transcripts.py --base_path PATH_TO_DATA --speaker SPEAKER_YOU_WANT`
`$ python match_transcript_gesture_timings.py --base_path PATH_TO_DATA --speaker SPEAKER_YOU_WANT`
baddabingbaddaboom, transcripts and matched up timings to muck about with. 
