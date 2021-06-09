import cv2
import numpy as np


IMG_HEIGHT = 900
ROW_HEIGHT = 50


def image_resize(image, width=None, height=None):
    """
    Resizes an image to one of the given dimensions while keeping the aspect ratio.
    """
    (h, w) = image.shape[:2]
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim)
    return resized


def split_img_into_rows(thresh, original):
    """
    Splits the image into lines using dilation and contours. Only returns
    rectangles from the original image with a minimum area. Sorts the row from
    top to bottom.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize=(100, 1))

    dilation = cv2.dilate(thresh, kernel, iterations=1)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    thresh_rows = []
    original_rows = []
    for c in reversed(contours):
        area = cv2.contourArea(c)
        if area > 2.5 * IMG_HEIGHT:
            x, y, w, h = cv2.boundingRect(c)
            thresh_rows.append(thresh[y : y + h, x : x + w])
            original_rows.append(original[y : y + h, x : x + w])

    return thresh_rows, original_rows


def split_rows_into_words(thresh_rows, original_rows):
    """
    Splits the lines into individual words using dilation and contours. Only returns
    rectangles from the original image with a minimum area. Sorts the words from
    left to right.
    """
    kernel = np.ones((5, 5), np.uint8)
    words = []

    for thresh_row, original_row in zip(thresh_rows, original_rows):
        thresh_row = image_resize(thresh_row, height=ROW_HEIGHT)
        original_row = image_resize(original_row, height=ROW_HEIGHT)

        dilation = cv2.dilate(thresh_row, kernel, iterations=2)
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        filtered = filter(lambda c: cv2.contourArea(c) > 10 * ROW_HEIGHT, contours)
        boxes = list(map(lambda c: cv2.boundingRect(c), filtered))

        for b in sorted(boxes, key=lambda b: b[0]):
            x, y, w, h = b
            words.append(original_row[y : y + h, x : x + w])

    return words


def preprocess(image, height):
    """
    Transforms an image of handwriting into a binary version so that it can be used to
    split the handwriting into individual words.
    """
    resized = image_resize(image, height=height)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (3, 3), cv2.BORDER_DEFAULT)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    return thresh, gray


def words_from_image(image):
    """
    Splits a received image of a paragraph of handwriting into individual lines, then
    splits the lines into individual words and prepares them to be inputted into
    the neural network.
    """
    thresh, gray = preprocess(image, IMG_HEIGHT)

    thresh_rows, original_rows = split_img_into_rows(thresh, gray)
    words = split_rows_into_words(thresh_rows, original_rows)

    return words


def words_from_line(line):
    """
    Splits a received image of a line of handwriting into individual words and prepares
    them to be inputted into the neural network.
    """
    thresh, gray = preprocess(line, ROW_HEIGHT)

    words = split_rows_into_words([thresh], [gray])
    return words


def transform_word(word):
    """
    Prepares a received image of a single word to be inputted into the neural network.
    """
    resized = image_resize(word, height=ROW_HEIGHT)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return [gray]


if __name__ == "__main__":
    img = cv2.imread("examples/paragraph.png")
    words = words_from_image(img)
    for word in words:
        cv2.destroyAllWindows()
        cv2.imshow("Word", word)
        cv2.waitKey(0)
