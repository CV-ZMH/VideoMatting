import os
import mimetypes

from moviepy.editor import VideoFileClip

def add_audio(src, dest):
    with VideoFileClip(src).audio as audio:
        with VideoFileClip(dest) as dest_clip:
            result_clip = dest_clip.set_audio(audio)
            os.remove(dest)
            result_clip.write_videofile(dest, audio_codec='libmp3lame', logger=None)

def get_extensions(file_type='image'):
    return set(k for k, v in mimetypes.types_map.items() if v.startswith(file_type + '/'))

def is_video(x):
    return os.path.splitext(x)[-1] in get_extensions('video')
