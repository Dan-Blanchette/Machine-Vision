#Author: Dan Blanchette
#Date: Sept 9th
#Assignemtnt 3 Camera Calibration

# Description: This program will calibrate my computer's webcam using
# the chessboard pattern and it will save a JSON file with the parameters
# required to achieve the same calibration with another image.

'''
Sources:
https://learnopencv.com/camera-calibration-using-opencv/
'''


import cv2 as cv
import numpy as np
import glob
import json

CHECKERBOARD = (6, 9)
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objpoints = []

imgpoints = []

objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1,2)
prev_img_shape = None

images = glob.glob('./images/*.png')

for fname in images:
   img = cv.imread(fname)
   gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

   ret, corners = cv.findChessboardCorners(gray, 
                                           CHECKERBOARD, 
                                           cv.CALIB_CB_FAST_CHECK + 
                                           cv.CALIB_CB_NORMALIZE_IMAGE)
   
   if ret == True:
      objpoints.append(objp)

      corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

      imgpoints.append(corners2)

      img = cv.drawChessboardCorners(img, CHECKERBOARD, corners2, ret)
   
   cv.imshow('img', img)
   cv.waitKey(0)

cv.destroyAllWindows()

h,w = img.shape[:2]

ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, 
                                                  imgpoints, 
                                                  gray.shape[::-1], 
                                                  None, 
                                                  None)



# create np array for json list conversion
jobjpoints = np.array(objpoints)
jimgpoints = np.array(imgpoints)
jgrayshape = np.array(gray.shape[::-1])
jdist = np.array(dist)
jmtx = np.array(mtx)
jret = np.array(ret)
jrvecs = np.array(rvecs)
jtvecs = np.array(tvecs)

# data to save for calibration via JSON format
output_objects = {
   "dist" : jdist.tolist(),
   "grayshape" : jgrayshape.tolist(),
   "mtx" : jmtx.tolist(),
   "objpoints" : jobjpoints.tolist(),
   "imgpoints" : jimgpoints.tolist(),
   "ret" : jret.tolist(),
   "rvecs" : jrvecs.tolist(),
   "tvecs" : jtvecs.tolist()
}

# export JSON with calibration settings
with open('camera_params.json', 'w') as fp:
   json.dump(output_objects, fp)

#TESTING TO SEE WHAT VALUES ARE GOING TO BE RECORDED

# print("Camera matrix : \n")
# print(mtx)
# print("dist : \n")
# print(dist)
# print("rvecs : \n")
# print(rvecs)
# print("tvecs : \n")
# print(tvecs)