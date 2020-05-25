import cv2

#read the images
im = cv2.imread('test.jpg')
h, w = im.shape[:2]
print (h, w)

#creating the gray scale version
gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

#OTHERS
# gray = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
# gray = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)

# print(gray)

#Compute the integral image
intim = cv2.integral(gray)

#normalize and save
intim = (255.0 * intim ) / intim.max()

#save the result in opencv window
cv2.imshow('test.jpg', im)
cv2.waitKey()

#write the images
cv2.imwrite('test.jpg', intim)
# print(intim)

