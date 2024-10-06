# Dan Blanchette
# Machine Vision
# Credits: https://sokacoding.medium.com/simple-motion-detection-with-python-and-opencv-for-beginners-cdd4579b2319
# Assignment 4: Motion Detector
'''This program will activate a webcam once
   motion is detected then place a bouding
   box around the source of the foreground
   motion. It will also record a video for
   240 frames then produce a .mp4 file.
'''

import cv2 as cv
import numpy as np



video_cap = cv.VideoCapture(0)

first_frame = video_cap.read()

result = 0
first_val = True
last_mean = 0
motion = False
frame_record_count = 0
# setup codecs and resolution for video recording
fourcc = cv.VideoWriter_fourcc(*'mp4v')
vid_file = cv.VideoWriter('output.mp4', fourcc, 20.0, (640, 480))

# first frame recorded used as a reference value for bouding box
prev_gray = cv.cvtColor(first_frame[1], cv.COLOR_BGR2GRAY)

key = ord('r')



while key != ord('s') or frame_record_count == 240:
   frame = video_cap.read()
   gray = cv.cvtColor(frame[1], cv.COLOR_BGR2GRAY)
   # Compute the absolute difference between current and previous frames
   frame_diff = cv.absdiff(prev_gray, gray)

   _, thresh = cv.threshold(frame_diff, 25, 255, cv.THRESH_BINARY)

   dilated = cv.dilate(thresh, None, iterations=2)

   non_zero_pixels = np.where(dilated > 0)
   if len(non_zero_pixels[0]) > 0 and len(non_zero_pixels[1]) > 0:
      x_min = np.min(non_zero_pixels[1])
      x_max = np.max(non_zero_pixels[1])
      y_min = np.min(non_zero_pixels[0])
      y_max = np.max(non_zero_pixels[0])

      # Shrink the bounding box by a percentage
      box_width = x_max - x_min
      box_height = y_max - y_min
      shrink_factor = 0.1  # 10% smaller bounding box
      shrink_x = int(box_width * shrink_factor / 2)
      shrink_y = int(box_height * shrink_factor / 2)

      x_min = max(0, x_min + shrink_x)
      y_min = max(0, y_min + shrink_y)
      x_max = min(frame[1].shape[1], x_max - shrink_x)
      y_max = min(frame[1].shape[0], y_max - shrink_y)
      # Draw the Box
      cv.rectangle(frame[1], (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

   # calculate the mean difference between
   # recent grayscaled frames and the one before it
   result = np.abs(np.mean(frame[1]) - last_mean)
   print(result)
   last_mean = np.mean(frame[1])



   # branching condition to not read
   # webcam initializing frame as a
   # false positive value for video
   # recording.
   if first_val:
      first_val = False
      pass
   else:
      # if webcam's calculated mean value exceeds 0.3 
      # (arbitrary depending on webcam performance 
      # and resolution)
      if result > 0.5:
         print("motion detected")
         motion = True
      if motion:
         print("started recording")
         cv.imshow("Motion Detected", frame[1])
         prev_gray = gray.copy()
         vid_file.write(frame[1])
         frame_record_count = frame_record_count + 1
   
   key = cv.waitKey(1)
   

   if (frame_record_count == 240):
      break

video_cap.release()
cv.destroyAllWindows()

