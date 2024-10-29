import cv2 as cv
import numpy as np


# refer to the last 10 mins of class for import library
webcam = cv.VideoCapture(0)



def bf_match():

   img1 = cv.imread('../sample_images/pup1.jpg', 0)
   img1 = cv.resize(img1, (int(img1.shape[1]*0.5), int(img1.shape[0]*0.5)))
   img2 = cv.imread('../sample_images/pup2.jpg', 0)
   img2 = cv.resize(img2, (int(img2.shape[1]*0.5), int(img2.shape[0]*0.5)))


   orb = cv.ORB_create()

   kp1, des1 = orb.detectAndCompute(img1, None)
   kp2, des2 = orb.detectAndCompute(img2, None)

   bf = cv.BFMatcher_create(cv.NORM_HAMMING)

   matches = bf.match(des1, des2)
   matches = sorted(matches, key=lambda x:x.distance)

   img3 = cv.drawMatches(img1,
                         kp1,
                         img2,
                         kp2,
                         matches[:20],
                         flags=2,
                         outImg=None)
   
   key = ord('r')
   while key != ord('s'):
      cv.imshow('bf_out', img3)
      key = cv.waitKey(1)

bf_match()