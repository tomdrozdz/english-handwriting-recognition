import cv2
import numpy as np


class ToGray:
    """Convert the image to grayscale."""

    def __call__(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return gray


class Blur:
    """Blur the image to smoothen out sharp edges."""

    def __init__(self, kernel=None):
        if kernel is None:
            self.kernel = (5, 5)
        else:
            self.kernel = kernel

    def __call__(self, img):
        blur = cv2.GaussianBlur(img, self.kernel, cv2.BORDER_DEFAULT)
        return blur


class Binarize:
    """Convert the image to black and white."""

    def __init__(self, white_text=False):
        if white_text:
            self.method = cv2.THRESH_BINARY_INV
        else:
            self.method = cv2.THRESH_BINARY

    def __call__(self, img):
        _, thresh = cv2.threshold(img, 0, 255, self.method + cv2.THRESH_OTSU)
        return thresh


class Resize:
    """
    Resize the image to set width and height while keeping the aspect ratio.
    The remaining space is filled with color white.
    """

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
        target = np.ones((self.height, self.width), dtype=np.uint8) * 255
        target[0 : new_size[1], 0 : new_size[0]] = img
        return target


class ToSinglularBatch:
    """Convert singular image to a batch of size one."""

    def __call__(self, img):
        batch = img.unsqueeze(0)
        return batch
