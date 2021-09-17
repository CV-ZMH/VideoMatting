import os.path as osp
from itertools import cycle

import cv2

from utils import ForegroundReader, BackgroundReader
from utils import utils

class MatteRunner:
    """Runner for video matte with MODNet"""
    def __init__(self, predictor, source, background, demo_mode=False):
        self.predictor = predictor
        self.fg_vid = ForegroundReader(source)
        self.bg_vid = BackgroundReader(background)
        self.demo_mode = demo_mode
        self.is_video = utils.is_video(source)
        self.count = 0

    def run(self, show=False):
        save_path = osp.splitext(self.fg_vid.source)[0] + '_result.mp4'
        # loop over foreground video and background video
        for fg, bg in zip(self.fg_vid, cycle(self.bg_vid)):
            matte = self.predictor.predict(fg)
            # blend foreground and background with predicted result
            res_img = self.blend(fg, bg, matte)
            # save the result
            if self.is_video:
                if self.fg_vid.frame_cnt == 1:
                    writer = self.fg_vid.get_writer(
                        res_img, save_path, self.fg_vid.FPS)
                writer.write(res_img)
            else:
                save_path = save_path[:-4] + '.png'
                cv2.imwrite(save_path, res_img)
            if show: # show the result
                key = self.fg_vid.show(res_img, 'matting result')
                if key == 27 or key == ord('q'):
                    break
        try:
            cv2.destroyAllWindows()
            if self.is_video:
                writer.release()
                utils.add_audio(self.fg_vid.source, save_path)
        except:
            exit

    def blend(self, fg, bg, matte):
        matte = cv2.resize(matte, (fg.shape[1], fg.shape[0]))
        bg = cv2.resize(bg, (fg.shape[1], fg.shape[0]))
        if self.demo_mode:
            fg, bg, matte = self._demo_animation(fg, bg, matte)
        res_img = (fg * matte + bg * (1 - matte))
        return res_img.astype('uint8')

    def _demo_animation(self, fg, bg, matte):
        """
        Demo mode for transformation effect according to self.count.
            0 ~ 30: target
            30 ~ 120: target to original (from L to R)
            120 ~ 150: original
            150 ~ 240: original to target (from R to L)
        """
        h, w = fg.shape[:2]
        if self.count < 120:
            if self.count  > 30:
                offset = int(w * (self.count - 120) / 59)
                # print(f"count {self.count} => {offset}")
                matte[:, 0:offset] = 1.0

        elif self.count < 240:
            if self.count < 150:
                matte[:, :] = 1.0
            else:
                offset = int(w * (self.count - 240) / 59)
                matte[:, 0:-offset] = 1.0
        self.count = (self.count + 1) % 240
        return fg, bg, matte
