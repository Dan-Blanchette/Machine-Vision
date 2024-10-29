import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt

# Dan Blanchette
# Machine Vision: Assignment #5 Feature Matching
# October 21, 2024

# References:
# openCV Documentation
# https://docs.opencv.org/4.x/dc/dc3/tutorial_py_matcher.html

MIN_MATCH_COUNT = 10

# image
master = cv.imread('imgs/cowboy_master.png', 0)
test_img1 = cv.imread('imgs/cowboy_test1.png', 0)
test_img2 = cv.imread('imgs/cowboy_test2.png', 0)
test_img3 = cv.imread('imgs/cowboy_test3.png', 0)

# ********** FUNCTIONS *****************
 
# uses orb feature detection
def orb(master_img, test_img):

   orb = cv.ORB_create()

   keyp1, des1 = orb.detectAndCompute(master_img, None)
   keyp2, des2 = orb.detectAndCompute(test_img, None)

   return keyp1, keyp2, des1, des2


# uses brute force KNN matching
def bruteForce(m_img, train_img, keyp1, keyp2, des1, des2):

   bf = cv.BFMatcher()

   matches = bf.knnMatch(des1, des2, k=2)

   good = []

   for m,n in matches:
      if m.distance < 0.85*n.distance:
         good.append([m])

   matched_img = cv.drawMatchesKnn(m_img, keyp1, train_img, keyp2, good, 
                                   None, flags=2)
   return matched_img

def orb_bf(master, test_img):
   keyp1, keyp2, des1, des2 = orb(master, test_img)
   results = bruteForce(master, test_img1, keyp1, keyp2, des1, des2)
   return results

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
      if m.distance < 0.7*n.distance:
         good.append([m])

   return good

def sift_bf(master, test_img):
   kp1, des1, kp2, des2 = sift(master, test_img)
   matches = bfMatcher(des1, des2) 
   good = ratioTest(matches)
   results = cv.drawMatchesKnn(master, kp1, test_img, kp2, good, None, 
                                flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)   
   return results

def flann(test_img, kp1, des1, kp2, des2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

            # store all the good matches as per Lowe's ratio test.
    good = []
    for m, n in matches:
        if m.distance < 0.7*n.distance:
            good.append(m)

    if len(good) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        h, w = master.shape
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]
                        ).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, M)

        test_img = cv.polylines(test_img, [np.int32(dst)], True, 255, 3, cv.LINE_AA)

    else:
        print("Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT))
        matchesMask = None

    return matchesMask, good

def draw_par(matchesMask):
   draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
               singlePointColor=None,
               matchesMask=matchesMask,  # draw only inliers
               flags=2)
   
   return draw_params

def sift_flann_homography(master, test_img):
   kp1, des1, kp2, des2 = sift(master, test_img)
   matchesMask, good = flann(test_img, kp1, des1, kp2, des2)
   draw_params = draw_par(matchesMask)
   results = cv.drawMatches(master, kp1, test_img, kp2, good, None, **draw_params)    

   return results

def fast_orb_bf(master, test_img):
   fast = cv.FastFeatureDetector_create()
   kp1 = fast.detect(master, None)
   kp2 = fast.detect(test_img, None)

   orb = cv.ORB_create()

   kp1, des1 = orb.compute(master, kp1)
   kp2, des2 = orb.compute(test_img, kp2)

   bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

   matches = bf.match(des1, des2)


   results = cv.drawMatches(master, kp1, test_img, kp2, matches[:50], None, 
                            flags=cv.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS,
                            matchesThickness=1
                            )

   return results




def main():
   # ----------------- #
   # ORB and BF Method #
   # ----------------- #
   results1 = orb_bf(master, test_img1)
   results2 = orb_bf(master, test_img2)
   results3 = orb_bf(master, test_img3)


   cv.imwrite("results/ORB BF Matching cowboy_test1.png", results1)
   cv.imwrite("results/ORB BF Matching cowboy_test2.png", results2)
   cv.imwrite("results/ORB BF Matching cowboy_test3.png", results3)

   # ------------------ #
   # SIFT and BF METHOD #
   # ------------------ #
   results1 = sift_bf(master, test_img1)
   results2 = sift_bf(master, test_img2)
   results3 = sift_bf(master, test_img3)
                                
   
   cv.imwrite("results/SIFT BF cowboy_test1.png", results1)
   cv.imwrite("results/SIFT BF cowboy_test2.png", results2)
   cv.imwrite("results/SIFT BF cowboy_test3.png", results3)

   # -------------------------------- #
   # SIFT FLANN and HOMOGRAPHY METHOD #
   # -------------------------------- #

   results1 = sift_flann_homography(master, test_img1)
   results2 = sift_flann_homography(master, test_img2)
   results3 = sift_flann_homography(master, test_img3)

    
   cv.imwrite("results/SIFT FLANN homography cowboy_test1.png", results1)
   cv.imwrite("results/SIFT FLANN homography cowboy_test2.png", results2)
   cv.imwrite("results/SIFT FLANN homography cowboy_test3.png", results3)

   # -------------------------------- #
   # FAST METHOD                      #
   # -------------------------------- #

   results1 = fast_orb_bf(master, test_img1)
   results2 = fast_orb_bf(master, test_img2)
   results3 = fast_orb_bf(master, test_img3)

    
   cv.imwrite("results/FAST BF cowboy_test1.png", results1)
   cv.imwrite("results/FAST BF cowboy_test2.png", results2)
   cv.imwrite("results/FAST BF cowboy_test3.png", results3)


if __name__ == '__main__':
   main()