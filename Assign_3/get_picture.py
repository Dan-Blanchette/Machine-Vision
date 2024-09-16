# Author: Dan Blanchette
# Date: September 9th 2024
# Assingment 3 camera calibration

# Descriptoin: This script will capture image test 
# patterns for the calibration script and store them in
# the images folder.

'''
Sources:
https://github.com/niconielsen32/CameraCalibration/blob/main/getImages.py

'''


import cv2 as cv

cap = cv.VideoCapture(0)

num = 0

# continue capturing pictures with the s key until escape is pressed
# or
while cap.isOpened():

    succes, img = cap.read()

    k = cv.waitKey(5)

    if k == 27:
        break
    elif k == ord('s'): # wait for 's' key to save and exit
        cv.imwrite('images/img' + str(num) + '_distorted.png', img)
        print("image saved!")
        num += 1

    cv.imshow('Img',img)

# Release and destroy all windows before termination
cap.release()

cv.destroyAllWindows()