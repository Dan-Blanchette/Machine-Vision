import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt


master = cv.imread('imgs/cowboy_master.png', cv.IMREAD_GRAYSCALE)  # queryImage



def sift(master, test_img):
   # Initiate SIFT detector
   sift = cv.SIFT_create()

   # find the keypoints and descriptors with SIFT
   kp1, des1 = sift.detectAndCompute(master, None)
   kp2, des2 = sift.detectAndCompute(test_img, None)
   return kp1, des1, kp2, des2

def bfMatcher(des1, des2):
   bf = cv.BFMatcher()
   matches = bf.knnMatch(des1, des2, k=2)
   return matches

def ratioTest(matches):
   good = []

   for m,n in matches:
      if m.distance < 0.75*n.distance:
         good.append([m])

   return good

def sift_bf(master, test_img):
   kp1, des1, kp2, des2 = sift(master, test_img)
   matches = bfMatcher(des1, des2) 
   good = ratioTest(matches)
   results = cv.drawMatchesKnn(master, kp1, test_img, kp2, good, None, 
                                flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)   
   return results
    

def main():
   test_img1 = cv.imread('imgs/cowboy_test1.png', cv.IMREAD_GRAYSCALE)  # trainImage
   test_img2 = cv.imread('imgs/cowboy_test2.png', cv.IMREAD_GRAYSCALE)  # trainImage
   test_img3 = cv.imread('imgs/cowboy_test3.png', cv.IMREAD_GRAYSCALE)  # trainImage

   results1 = sift_bf(master, test_img1)
   results2 = sift_bf(master, test_img2)
   results3 = sift_bf(master, test_img3)
                                
   
   cv.imwrite("results/SIFT BF cowboy_test1.png", results1)
   cv.imwrite("results/SIFT BF cowboy_test2.png", results2)
   cv.imwrite("results/SIFT BF cowboy_test3.png", results3)


if __name__ == '__main__':
    main()