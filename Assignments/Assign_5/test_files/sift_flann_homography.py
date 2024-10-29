import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

MIN_MATCH_COUNT = 10

master = cv.imread('imgs/cowboy_master.png', 0)  # queryImage
test_img1 = cv.imread('imgs/cowboy_test1.png', 0)
test_img2 = cv.imread('imgs/cowboy_test2.png', 0)
test_img3 = cv.imread('imgs/cowboy_test3.png', 0)



def sift(master, test_img):
    # Initiate SIFT detector
    sift = cv.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(master, None)
    kp2, des2 = sift.detectAndCompute(test_img, None)
    return kp1, des1, kp2, des2

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
    matchesMask1, good1 = flann(test_img, kp1, des1, kp2, des2)
    draw_params = draw_par(matchesMask1)
    results = cv.drawMatches(master, kp1, test_img, kp2, good1, None, **draw_params)    

    return results

def main():

    results1 = sift_flann_homography(master, test_img1)
    results2 = sift_flann_homography(master, test_img2)
    results3 = sift_flann_homography(master, test_img3)

    
    cv.imwrite("results/SIFT FLANN homography cowboy_test1.png", results1)
    cv.imwrite("results/SIFT FLANN homography cowboy_test2.png", results2)
    cv.imwrite("results/SIFT FLANN homography cowboy_test3.png", results3)


if __name__ == '__main__':
    main()