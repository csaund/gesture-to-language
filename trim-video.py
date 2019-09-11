from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

ffmpeg_extract_subclip("megyn-kelly.mp4", 6708, 6950, targetname="test.mp4")
