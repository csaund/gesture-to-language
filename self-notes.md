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
