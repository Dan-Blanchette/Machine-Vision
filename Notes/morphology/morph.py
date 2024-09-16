# Morphology Code example

import cv2 as cv
import numpy as np

# img = cv.imread('../../CS455-555_Machine_Vision_Student/examples/sample_images/octo1.jpg')

webcam = cv.VideoCapture(0)

# def nothing(x):
#    pass

def morph_ops():

   cv.namedWindow('controls')
   cv.createTrackbar('threshold', 'controls', 0, 255, lambda*args: None)
   cv.setTrackbarPos('threshold', 'controls', 127)

   
# img = np.zeros((300, 512, 3), np.uint8)

# cv.namedWindow('image')

# cv.createTrackbar('R', 'image', 0, 255, nothing)
# cv.createTrackbar('G', 'image', 0, 255, nothing)
# cv.createTrackbar('B', 'image', 0, 255, nothing)

# while True:
#    # show image
#    cv.imshow('image', img)

   # # for button pressing and changing
   # k = cv.waitKey(1) & 0xFF
   # if k == 27:
   #    break
   # r = cv.getTrackbarPos('R', 'image')
   # g = cv.getTrackbarPos('G', 'image')
   # b = cv.getTrackbarPos('B', 'image')

   # display color mixture
   # img[:] = [b, g, r]


key = ord('r')
morph_ops()

while key != ord('s'):
   still = webcam.read()
   img_gray = cv.cvtColor(still[1], cv.COLOR_BGR2GRAY)

   # ops
   thresh = int(cv.getTrackbarPos("threshold", "controls"))
   ret, img = cv.threshold(img_gray, thresh, 255, cv.THRESH_BINARY)

   cv.imshow('output img', img)
   key = cv.waitKey(5)


cv.destroyAllWindows()