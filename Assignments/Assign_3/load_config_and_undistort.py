# Author: Dan Blanchette
# Date: September 12th 2024
# Assingment 3 Camera Calibration

# Descriptoin: This script will get the picutre matrix and distance values 
# from a JSON file and use these values to undistort an as image.

'''
Sources: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
'''

import json
import cv2 as cv
import numpy as np

# laod JSON data
with open('camera_params.json', 'r') as fp:
   data = json.load(fp)


# convert to np array
loaded_matrix = data["mtx"]
np_matrix = np.array(loaded_matrix)
loaded_distance = data["dist"]
np_distance = np.array(loaded_distance)


# read a distorted iamge
img_distorted = cv.imread('images/img2_distorted.png')
h, w = img_distorted.shape[:2]
# use JSON values to and image matrix to pre-process the distorted image
newcameramtx, roi = cv.getOptimalNewCameraMatrix(np_matrix, np_distance, (w,h), 1, (w,h))

# undistort the image
dst = cv.undistort(img_distorted, np_matrix, np_distance, None, newcameramtx)

# crop the image
x, y, w, h = roi
dst = dst[y:y+h, x:x+w]
cv.imwrite('images/img2_undistorted_img.png', dst)