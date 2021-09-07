import os
import os.path as osp
import argparse
from itertools import cycle

import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from fire import Fire
# from moviepy.video.io.VideoFileClip import VideoFileClip

from predictor import Predictor
from utils.background import Background

import myutils
import myvideo


class MatteRunner:
    def __init__(self, predictor, fg_vid, bg_vid, demo_mode=False):
        self.predictor = predictor
        self.fg_vid = fg_vid
        self.bg_vid = bg_vid
        self.demo_mode = demo_mode

    def run(self, show=True):
        for fg, bg in zip(self.fg_vid, cycle(self.bg_vid)):
            matte = self.predictor.predict(fg)
            res_img = self.blend(fg, bg, matte)

            if self.fg_vid.frame_cnt == 1:
                save_path = osp.splitext(self.fg_vid.source)[0] + f'{TEMP}.mp4'
                writer = self.fg_vid.get_writer(
                    res_img, save_path, fps=30
                )

            writer.write(res_img)
            if show:
                key = self.fg_vid.show(res_img, 'matted_image')
                if key == 27 or key == ord('q'):
                    break

        writer.release()
        cv2.destroyAllWindows()

    def blend(self, fg, bg, matte):
        matte = cv2.resize(matte, (fg.shape[1], fg.shape[0]))
        bg = cv2.resize(bg, (fg.shape[1], fg.shape[0]))
        res_img = (fg * matte + bg * (1 - matte))
        return res_img.astype(np.uint8)


def main(source, background, weight='weights/modnet_mobilenetv2.pth'):
    fg_vid = myvideo.Video(vid_path)
    bg_vid = Background(bg_path)
    predictor = Predictor(weight_path)
    runner = MatteRunner(predictor, fg_vid, bg_vid, demo_mode=False)
    runner.run(show=True)


if __name__ == '__main__':
    TEMP = '_matted_black' 
    Fire(main)

    # matte_file = osp.splitext(vid_path)[0] + f'{TEMP}.mp4'
    # from moviepy.editor import VideoFileClip, AudioFileClip
    # audio_clip = AudioFileClip(vid_path)
    # matte_clip = VideoFileClip(matte_file)
    # videoclip2 = matte_clip.set_audio(audio_clip)

    # # videoclip2.preview(fps=15, audio=True)
    # videoclip2.write_videofile('movie.mp4')
