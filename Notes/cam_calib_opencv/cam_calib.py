import cv2 as cv

'''
img = cv.imread('./sample_images/ocoto1.jpg')

key = ord('r')

while key != ord('s'):
   cv.imshow("Octopus", img)
   cv.waitKey()

# cv.findChessboardCorners(image, (x_size, y_size), None)
cv.findChessboardCorners(img, (8.5, 11), None)
'''


'''
webcam = cv.VideoCapture(0)
key = ord('r')
while key != ord('s'):
   still = webcam.read()
   print(still)
   cv.imshow("Webcam", still[1])
   key = cv.waitKey(10)

cv.destroyAllWindows()
'''

# Pixel Values (pixel indexing)

# img = cv.imread('./sample/trex.jpg')
# # give me the pixel at 40x and 40y from channel 0
# print(img[40, 40, 0])

# print(img.shape)

# print(img.dtype)

# Image Borders
#

img = cv.imread('./sample_images/ocoto1.jpg')
img = img[250:500, 400:700] # octopus head

boarder_size = 10

# BRG channels
boarder_color = [255, 0, 0]

# img = cv.copyMakeBorder(img, 
#                         boarder_size, 
#                         boarder_size, 
#                         boarder_size, 
#                         boarder_size,
#                         cv.BORDER_CONSTANT,
#                         value=boarder_color)

# smears border
# img = cv.copyMakeBorder(img, 
#                         boarder_size, 
#                         boarder_size, 
#                         boarder_size, 
#                         boarder_size,
#                         cv.BORDER_REPLICATE,
#                         value=boarder_color)

# img = cv.copyMakeBorder(img, 
#                         boarder_size, 
#                         boarder_size, 
#                         boarder_size, 
#                         boarder_size,
#                         cv.BORDER_REFLECT,
#                         value=boarder_color)


# img = cv.copyMakeBorder(img, 
#                         boarder_size, 
#                         boarder_size, 
#                         boarder_size, 
#                         boarder_size,
#                         cv.BORDER_WRAP,
#                         value=boarder_color)

# key = ord('r')
# while key != 's':
#    cv.imshow('Bordered Octopus', img)
#    key = cv.waitKey()

# cv.destroyAllWindows()

# path = './output_images/octo_gw.png'
# cv.imwrite(path,img)

def write(img, text, org(50,50), font=cv.FONT_HERSHEY_SIMPLEX, fontScale=1,  color=(255,0,0), thickness=2, line_type=cv.LINE_AA):
   print("hello")
