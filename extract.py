import cv2
import numpy as np


IMG_HEIGHT = 900
ROW_HEIGHT = 50


def image_resize(image, width=None, height=None):
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
    kernel = np.ones((5, 5), np.uint8)
    words = []

    for thresh_row, original_row in zip(thresh_rows, original_rows):
        thresh_row = image_resize(thresh_row, height=50)
        original_row = image_resize(original_row, height=50)

        dilation = cv2.dilate(thresh_row, kernel, iterations=2)
        closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for c in contours:
            print(cv2.contourArea(c))

        filtered = filter(lambda c: cv2.contourArea(c) > 10 * ROW_HEIGHT, contours)
        boxes = list(map(lambda c: cv2.boundingRect(c), filtered))

        for b in sorted(boxes, key=lambda b: b[0]):
            x, y, w, h = b
            words.append(original_row[y : y + h, x : x + w])

    for word in words:
        cv2.destroyAllWindows()
        cv2.imshow("Word", word)
        cv2.waitKey(0)

    return words


def words_from_image(image):
    resized = image_resize(image, height=IMG_HEIGHT)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (3, 3), cv2.BORDER_DEFAULT)
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    thresh_rows, original_rows = split_img_into_rows(thresh, gray)
    words = split_rows_into_words(thresh_rows, original_rows)

    return words


if __name__ == "__main__":
    words_from_image(cv2.imread("real1.jpg"))
