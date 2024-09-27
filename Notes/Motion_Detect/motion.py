import cv2 as cv
import numpy as np


img = cv.imread("../sample_images/")

webcam = cv.VideoCapture(0)

def background_sub():
   bg = cv.createBackgroundSubtractorMOG2()

   key = ord('r')

   while key != ord('s'):
      still = webcam.read()
      img_current_gray = cv.cvtColor(still[1], cv.COLOR_BGR2GRAY)
      img = bg.apply(img_current_gray)
      cv.imshow('output', img)
      key = cv.waitKey(5)


# background_sub()

def contour_masking():
   cv.namedWindow('controls')

   cv.createTrackbar('lower', 'controls', 0, 255, lambda *args: None)
   cv.createTrackbar('upper', 'controls', 0, 255, lambda *args: None)
   key = ord('r')

   while key != ord('s'):
      still = webcam.read()
      img = still[1].copy()
      img_current_gray = cv.cvtColor(still[1], cv.COLOR_BGR2GRAY)
      img_blur = cv.GaussianBlur(img_current_gray, (7,7), 0)

      lower = int(cv.getTrackbarPos('lower', 'controls'))
      upper = int(cv.getTrackbarPos('upper', 'controls'))

      img_canny = cv.Canny(img_blur, lower, upper)
      img_morphed = cv.morphologyEx(img_canny, cv.MORPH_CLOSE, np.ones((5,5)))

      contours, hierarchy = cv.findContours(img_morphed, 
                                            cv.RETR_EXTERNAL, 
                                            cv.CHAIN_APPROX_SIMPLE)
      
      contours = sorted(contours, key=cv.contourArea)
      mask = np.zeros_like(img_morphed)

      cv.drawContours(mask,
                      [contours[-1]],
                      -1,
                      255,
                      cv.FILLED,
                      1)
      img[mask==0] = 0
      cv.imshow('image', img)
      key = cv.waitKey(5)

contour_masking()
