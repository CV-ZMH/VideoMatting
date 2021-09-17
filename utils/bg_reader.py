import mimetypes
import cv2
import numpy as np

def get_extensions(type='image'):
    return tuple(k for k, v in mimetypes.types_map.items()
                 if v.startswith(type + '/'))

class BackgroundReader:
    def __init__(self, src):
        self.src = src
        if src is None:
            self.is_video = False
            self.bg_img = np.zeros((100, 100, 3), np.uint8)
        elif src.endswith(get_extensions('video')):
            self.is_video = True
            self.cap = cv2.VideoCapture(src)
            assert self.cap.isOpened(), f"video can't open {src}"
        else:
            self.is_video = False
            self.bg_img = cv2.imread(src)

    def __iter__(self):
        while True:
            if self.is_video:
                ret, img = self.cap.read()
                if not ret:
                    self.cap.release()
                    self.cap = cv2.VideoCapture(self.src)
                    _, img = self.cap.read()
                yield img
            else:
                yield self.bg_img.copy()

    def __del__(self):
        if self.is_video:
            try:
                self.cap.release()
            except:
                pass
