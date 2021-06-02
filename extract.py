import cv2
import numpy as np


HEIGHT = 900


def image_resize(image, height):
    (h, w) = image.shape[:2]
    r = height / h
    dim = (int(w * r), height)
    resized = cv2.resize(image, dim)
    return resized


img = cv2.imread("test1.png")
resized = image_resize(img, HEIGHT)
gray = cv2.cvtColor(resized, cv2.COLOR_RGB2GRAY)

blur = cv2.GaussianBlur(gray, (3, 3), cv2.BORDER_DEFAULT)
cv2.imshow("Blur", blur)
cv2.waitKey(0)

ret, thresh = cv2.threshold(blur, 210, 255, cv2.THRESH_BINARY_INV)
cv2.imshow("Threshold", thresh)
cv2.waitKey(0)

kernel = np.ones((5, 5), np.uint8)

dilation = cv2.dilate(thresh, kernel, iterations=2)
cv2.imshow("Dilation", dilation)
cv2.waitKey(0)

closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel)
cv2.imshow("Closing", closing)
cv2.waitKey(0)

contours, hierarchy = cv2.findContours(
    closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
)

to_display = resized

for c in contours:
    area = cv2.contourArea(c)
    if area > 0.5 * HEIGHT:
        x, y, w, h = cv2.boundingRect(c)
        to_display = cv2.rectangle(to_display, (x, y), (x + w, y + h), (255, 0, 0), 2)

cv2.imshow("Final", to_display)
cv2.waitKey(0)
