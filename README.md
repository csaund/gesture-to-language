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



## What I want
:white_check_mark: A script that takes a long video
:white_circle: Segments that video into gesture phrases and sub-gestures
:white_circle: Takes those mini-gesture videos, and classifies the gesture by movement characteristics
:white_check_mark: Takes those mini-gesture videos, and gets the transcript for them
:white_circle: Takes those transcripts, and gets their linguistic forms (nltk?)
:white_circle: Takes those linguistic forms, categorize them into linguistic categories


## Evaluation
* Computationally, see overlap between sentence categories and gesture categories
* Subjectively, see people's matching abilities (gesture --> candidate sentences from categories)
