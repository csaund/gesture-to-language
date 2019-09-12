# gesture-to-language


Alright so here we go the first thing's first is you can cut a video using this command:
`ffmpeg -i megyn-kelly.mp4 -ss 00:06:07 -to 00:06:09 -c:v copy -c:a copy test.mp4`

Then you can turn that into a wav that you can get the transcript for like this:
`ffmpeg -i test.mp4 -ab 160k -ac 2 -ar 48000 -vn audio_test.wav`

Then you can get the transcript for that like this:
`sudo python transcribe_word_time_offsets.py audio_test.wav`

And that will put the output into a file called `audio_test.json`

Rad.

From there I need to do wordnet things. Time to install wordnet I guess!




## What I want
* A script that takes a long video
* Segments that video into gestures
* Takes those mini-gesture videos, and classifies the gesture by movement characteristics
* Takes those mini-gesture videos, and gets the transcript for them
* Takes those transcripts, and gets their WN forms
* Takes those WN forms, categorize them into linguistic categories


## Evaluation
* Computationally, see overlap between sentence categories and gesture categories
* Subjectively, see people's matching abilities (gesture --> candidate sentences)
