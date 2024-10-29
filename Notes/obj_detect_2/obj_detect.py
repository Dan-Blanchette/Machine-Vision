# Object Detection Example

import cv2 as cv
from operator import index
import numpy as np
from cv2 import xfeatures2d

webcam = cv.VideoCapture(0)


def preprocess(img):
   img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
   img = cv.GaussianBlur(img, (7,7), 0)
   return img

# BROKEN
def dense():
   still = webcam.read()

   prevs = preprocess(still[1])

   hsv = np.zeros_like(still[1])

   hsv[..., 1] = 255

   print(hsv.shape)

   key = ord('r')

   while key != ord('s'):
      still = webcam.read()
      next = preprocess(still[1])

      flow = cv.calcOpticalFlowFarneback(prevs, 
                                         next, 
                                         None, 
                                         0.5, 
                                         3, 
                                         15, 
                                         3,
                                         5, 
                                         1.2, 
                                         0)
      
      mag, ang = cv.cartToPolar(flow[...,0], flow[...,1])
      hsv[...,0] = ang*180/np.pi/2
      hsv[...,2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)

      img = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

      cv.imshow('optical- flow-dense', img)
      key = cv.waitKey(5)
      prevs = next

# dense()


# Fixed
def harris():
   key = ord('r')

   while key != ord('s'):
      still = webcam.read()

      img = preprocess(still[1])

      img = np.float32(img)
      img = cv.cornerHarris(img, 2, 3, 0.1)
      img = cv.dilate(img, None)
      # threshold
      still[1][img>0.01*img.max()] =  [0,0,255]
      cv.imshow('harris output', img)
      key = cv.waitKey(5)

# harris()

def sift():
   key = ord('r')

   while key != ord('s'):

      still = webcam.read()

      img = preprocess(still[1])
      s_obj = cv.SIFT_create()
      kp = s_obj.detect(img, None)

      img = cv.drawKeypoints(still[1], kp, outImage=None)
      cv.imshow('sift output', img)
      key = cv.waitKey(5)

# sift()

def fast():
   key = ord('r')

   while key != ord('s'):

      still = webcam.read()

      img = preprocess(still[1])
      f_obj = cv.FastFeatureDetector_create()
      kp = f_obj.detect(img, None)

      b_obj = cv.xfeatures2d.BriefDescriptorExtractor_create()
      kp, des = b_obj.compute(img, kp)

      img = cv.drawKeypoints(still[1], kp, outImage=None)
      cv.imshow('fast output', img)
      key = cv.waitKey(5)

# fast()

def orb():
   key = ord('r')

   while key != ord('s'):

      still = webcam.read()

      img = preprocess(still[1])
      o_obj = cv.ORB_create()
      kp = o_obj.detect(img, None)

      b_obj = cv.xfeatures2d.BriefDescriptorExtractor_create()

      img = cv.drawKeypoints(still[1], kp, outImage=None, color=(0,255,0), flags=0)
      cv.imshow('orb output', img)
      key = cv.waitKey(5)
orb()