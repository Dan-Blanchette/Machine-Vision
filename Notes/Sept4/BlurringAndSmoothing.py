import cv2 as cv
import numpy as np


# Gaussian Blur example

img = cv.imread('sample_images/octo1.jpg')

# smoothing  average
blurred = cv.blur(img, (15,15), 0)

# gaussian blur
gauss_blur = cv.GaussianBlur(img, (3.3), 0)

#median blur
med_blur = cv.medianBlur(img, 15)

# bilaterial filtering
blurred = cv.bilateralFilter(img, 11, 61, 39)

# Non-Local Means [NOTE: Computationally Expensive]

cv.fastNlMeansDenoising(img)
cv.fastNlMeansDenoisingColored(img)
cv.fastNlMeansDenoisingMulti(img)
cv.fastNlMeansDenoisingMultiColored(img)