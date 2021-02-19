import numpy as np
import cv2
import os

os.system("cls")

#path = "C:\\Users\\INKOM06\\Pictures\\handwash\\mod1\\"

path =r"C:\Users\INKOM06\Pictures\_DATASET\roadcibi"

files = os.listdir(path)
fileIdx = 0

frame = cv2.imread(os.path.join(path,files[fileIdx]))

M = frame.shape[0]
N = frame.shape[1]
ratio = 0.2

frame = cv2.resize(frame,(int(ratio*N), int(ratio*M)) , interpolation = cv2.INTER_AREA)   
filterSz = 5
frame = cv2.blur(frame,(filterSz,filterSz))







labImg = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
hsvImg = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)



[Blue,G,R] = cv2.split(frame)
[L,A,B] = cv2.split(labImg)
[H,S,V] = cv2.split(hsvImg)



clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
#H = clahe.apply(H)




oriMerge = np.hstack((frame,frame,frame))
rgbMerge = np.hstack((R,G,Blue))
labMerge = np.hstack((L,A,B))
hsvMerge = np.hstack((H,S,V))

finMerge = np.vstack((rgbMerge,labMerge, hsvMerge))


cv2.imshow("finMerge", finMerge)


ret,thH = cv2.threshold(H,50,255,cv2.THRESH_BINARY_INV)

#cv2.imshow("Hue", H)

#cv2.imshow("Binary image", thH)


cv2.waitKey(0)
cv2.destroyAllWindows()

