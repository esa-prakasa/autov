import cv2
import os
from imutils.object_detection import non_max_suppression
from imutils import resize
import numpy as np


os.system("cls")

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

img = cv2.imread(r'C:\Users\Esa\Documents\a1ocvCodes\auto8\pedest3.jpg')
img = resize(img,height=800)

rects,weights = hog.detectMultiScale(img,winStride=(4,4),padding=(8,8),scale=1.05)
# rects,weights = hog.detectMultiScale(img,winStride=(2,2),padding=(8,8),scale=2.05)

copy = img.copy()
for x,y,w,h in rects:
    cv2.rectangle(copy,(x,y),(x+w,y+h),(0,0,255),1)

cv2.imshow('before suppression',copy)
cv2.waitKey(0)

r = np.array([[x,y,x+w,y+h] for x,y,w,h in rects])
pick = non_max_suppression(r,probs=None,overlapThresh=0.65)    

for xa,ya,xb,yb in pick:
    cv2.rectangle(img,(xa,ya),(xb,yb),(0,255,0),2)

cv2.imshow('after suppression',img)
cv2.waitKey(0)

cv2.destroyAllWindows()