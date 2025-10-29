# this code should be run under "spleeter" environment
# package spleeter can only work on python<3.11, which is a conflict of other packages

import spleeter.separator
import moviepy.editor as mp
import os
from tqdm import tqdm

separator = spleeter.separator.Separator('spleeter:2stems')

def separate(video_path : str = None, audio_path : str = None, output_path : str = "../dataset/audios") :
    global separator
    if audio_path is None :
        video = mp.VideoFileClip(video_path)
        audio_path = video_path[: -4] + ".mp3"
        video.audio.write_audiofile(audio_path, logger=None)
        os.remove(video_path)
        if output_path is None :
            output_path = video[: -4]
    elif output_path is None :
        output_path = audio_path[: -4]
    separator.separate_to_file(audio_path, output_path)


if __name__ == "__main__" :
    items = os.listdir("../dataset/resources")
    for item in tqdm(items) :
        if item.endswith(".mp4") :
            separate(video_path=os.path.join("../dataset/resources", item))
        else :
            separate(audio_path=os.path.join("../dataset/resources", item))