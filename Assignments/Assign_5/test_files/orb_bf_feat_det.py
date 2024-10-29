import cv2 as cv
import numpy as np

master = cv.imread('imgs/cowboy_master.png', 0)
test_img1 = cv.imread('imgs/cowboy_test1.png', 0)
test_img2 = cv.imread('imgs/cowboy_test2.png', 0)
test_img3 = cv.imread('imgs/cowboy_test3.png', 0)



# uses orb feature detection
def orb(master_img, test_img):

   orb = cv.ORB_create()

   keyp1, des1 = orb.detectAndCompute(master_img, None)
   keyp2, des2 = orb.detectAndCompute(test_img, None)

   # orb keypoint visualization
   # imgKp1 = cv.drawKeypoints(master_img, keyp1, None)
   # imgKp2 = cv.drawKeypoints(test_img, keyp2, None)

   return keyp1, keyp2, des1, des2


# uses brute force KNN matching
def bruteForce(m_img, train_img, keyp1, keyp2, des1, des2):

   bf = cv.BFMatcher()

   matches = bf.knnMatch(des1, des2, k=2)

   good = []

   for m,n in matches:
      if m.distance < 0.85*n.distance:
         good.append([m])

   matched_img = cv.drawMatchesKnn(m_img, keyp1, train_img, keyp2, good, None, flags=2)
   return matched_img

def orb_bf(master, test_img):
   keyp1, keyp2, des1, des2 = orb(master, test_img)
   results = bruteForce(master, test_img1, keyp1, keyp2, des1, des2)
   return results

def main():

   # ORB and BF Method
   results1 = orb_bf(master, test_img1)
   results2 = orb_bf(master, test_img2)
   results3 = orb_bf(master, test_img3)


   cv.imwrite("results/ORB BF Matching cowboy_test1.png", results1)
   cv.imwrite("results/ORB BF Matching cowboy_test2.png", results2)
   cv.imwrite("results/ORB BF Matching cowboy_test3.png", results3)


if __name__ == '__main__':
   main()


