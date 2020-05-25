import cv2
from numpy import zeros
# import numpy as np
from numpy import uint8
from numpy import ones

#read the image
im = cv2.imread('temple.jpg')
# h, w = im.shape[:2]
# # print (h, w)
# #flood fill example
# diff = (6, 6, 6)
# mask = zeros((h+2, w+2), np.uint8)
# cv2.floodFill(im, mask, (10,10), (255, 255, 0), diff, diff)

#SURF Features extractced and plotted using OpenCV
im_lowres = cv2.pyrDown(im)

#convert into gray scale
gray = cv2.cvtColor(im_lowres, cv2.COLOR_RGB2GRAY)

#Detect the feature points
s = cv2.xfeatures2d.SURF_create()
mask = uint8(ones(gray.shape))
keypoints = s.detect(gray, mask)

#show the image and plots
vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

for k in keypoints[::10]:
    cv2.circle(vis,(int(k.pt[0]),int(k.pt[1])),2,(0,255,0),-1)
    cv2.circle(vis,(int(k.pt[0]),int(k.pt[1])),int(k.size),(0,255,0),2)   
#open the image in opencv windows
cv2.imshow('local descriptors', vis)
cv2.waitKey()

cv2.imwrite('SURF1.jpg', im)