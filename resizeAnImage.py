import os
import cv2
import numpy as np
import math

os.system("cls")


inputPath  = r'C:\Users\INKOM06\Pictures\roadDataset\oregon_us\roads_annotated\ds\bw'
colPath = r'C:\Users\INKOM06\Pictures\roadDataset\oregon_us\roads_annotated\ds\img'




ratio = 0.3

fileList = os.listdir(inputPath)
filePath = os.path.join(inputPath,fileList[0])
img = cv2.imread(filePath)
img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imshow("ori",img)

M = img.shape[0]
N = img.shape[1]

m = math.floor(M*ratio)
n = math.floor(N*ratio)

img = cv2.resize(img, (n,m), interpolation = cv2.INTER_AREA)
cv2.imshow("resize",img)



fileList = os.listdir(colPath)
filePath = os.path.join(colPath,fileList[0])
imgRGB = cv2.imread(filePath)
#imgRGB = cv2.cvtColor(imgRGB, cv2.COLOR_BGR2GRAY)

cv2.imshow("ori RGB",imgRGB)

M = imgRGB.shape[0]
N = imgRGB.shape[1]

m = math.floor(M*ratio)
n = math.floor(N*ratio)

imgRGB = cv2.resize(imgRGB, (n,m), interpolation = cv2.INTER_AREA)
cv2.imshow("resize RGB",imgRGB)


cv2.waitKey(0)
cv2.destroyAllWindows()
