import cv2 as cv
import numpy as np


# Gaussian Blur example

img = cv.imread('sample_images/octo1.jpg')

blurred = cv.GaussianBlur(img, (3,3), 1)