Gesture Data from analyze_gestures looks like this:
[
  {
      id: 748392
      keyframes: [
        x: [...]
        y: [...]
      ],
      [
        x: [...]
        y: [...]
      ]
  },
  ...
]

so the keyframes are organized by frame, `[t1, t1', t1'', ...], [t2, t2', t2''...]...`.
Use  `arrange_data_by_time` to get `[t1, t2,...], [t1', t2',...], ...` so you can plot easily.

Mostly the thing I do right when I start a new session is
```
$ from analyze_frames import *
$ video_base_path = "/Users/carolynsaund/github/gest-data/data/rock/keypoints_simple/"
$ timings_path = "/Users/carolynsaund/github/gest-data/data/rock/timings.json"
$ asgd = analyze_gestures(video_base_path, timings_path)
```
This gives all the gestures loaded and ready to play with but takes forever. Easier to get a
single gesture by timings,
```
$ id = 76398
$ get_single_gesture_from_timings(id, video_base_path, timings_path)
```
to play with single gesture data.


Remember to delete gesture-to-lang-copy



TODO
- figure out gesture cluster <--> sentence cluster mapping -- how to do?
- see which feature is most influential in clustering
- limit number of clusters
- check distance for clusters with only 1 feature
- manually make some features more/less important
- report importance of each individual feature in clustering
- try seeding with some gestures? -- bias, but since we're doing it based on prior gesture research it's ok

    # TODO check what audio features are in original paper

    ## Audio for agent is bad -- TTS is garbaggio
    ## assumes there is good prosody in voice (TTS there isn't)

    ## Co-optimize gesture-language clustering (learn how)
    ## KL distance for two clusters?

- Frame intro of paper

- learn similarity of sentences from within one gesture
    # how to map gesture clusters <--> sentence clusters
    ## in the end want to optimize overlapping clusters btw gesture/language

    # probabilistic mapping of sentence (from gesture) to sentence cluster
