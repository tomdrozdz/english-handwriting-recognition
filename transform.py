import cv2
import numpy as np


class ToGray:
    def __call__(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray


class Blur:
    def __init__(self, kernel=None):
        if kernel is None:
            self.kernel = (5, 5)
        else:
            self.kernel = kernel

    def __call__(self, img):
        blur = cv2.GaussianBlur(img, self.kernel, cv2.BORDER_DEFAULT)
        return blur


class Binarize:
    def __init__(self, white_text=False):
        if white_text:
            self.method = cv2.THRESH_BINARY_INV
        else:
            self.method = cv2.THRESH_BINARY

    def __call__(self, img):
        _, thresh = cv2.threshold(img, 0, 255, self.method + cv2.THRESH_OTSU)
        return thresh


class Resize:
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def __call__(self, img):
        h, w = img.shape[:2]
        fx = w / self.width
        fy = h / self.height
        f = max(fx, fy)

        new_size = (
            max(min(self.width, int(w / f)), 1),
            max(min(self.height, int(h / f)), 1),
        )
        img = cv2.resize(img, new_size)
        target = np.ones((self.height, self.width)) * 255
        target[0 : new_size[1], 0 : new_size[0]] = img
        return target
