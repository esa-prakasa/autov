import cv2
import os
import numpy as np


# confusion matrix in sklearn
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

os.system('cls')

rootPath = r'C:\Users\Esa\Pictures\_DATASET\unetpusbin\topCamSegmented\kFoldFolders\D3'   ## fold1 Train: D0-D1-D2  Test: D3
resPath = r'C:\Users\Esa\Pictures\_DATASET\unetpusbin\result_fold0'

folderNames = os.listdir(rootPath)
resNames = os.listdir(resPath)

idx= int(input("What is the index? "))
fullPath0 = rootPath+"\\"+folderNames[idx]+"\\mask\\" #_mask.png"

print(fullPath0)

maskFiles = os.listdir(fullPath0)

fullPath0 = fullPath0+maskFiles[0]
print(fullPath0)


fullPath1 = resPath+"\\"+ folderNames[idx]+".png" #resNames[idx]
print(fullPath1)
img0 = cv2.imread(fullPath0)
M = int(0.5*img0.shape[0])
N = int(0.5*img0.shape[1])

img0 = cv2.resize(img0, (N,M), interpolation = cv2.INTER_AREA)

img1 = cv2.imread(fullPath1)
img1 = cv2.resize(img1, (N,M), interpolation = cv2.INTER_AREA)

imgF = np.hstack((img0,img1))
cv2.imshow(str(idx),imgF)


imgAND = np.zeros((M,N), np.uint8)
imgOR = np.zeros((M,N), np.uint8)

sumAND = 0
sumOR  = 0
for i in range(M):
    for j in range(N):
        im0v = img0[i,j,0]
        im1v = img1[i,j,0]

        if (im0v==255) and (im1v==255):
            imgAND[i,j]=255
            sumAND +=1

        if (im0v==255) or (im1v==255):            
            imgOR[i,j]=255
            sumOR +=1

imgANDOR = np.hstack((imgAND,imgOR))

cv2.imshow("AND and OR Operations", imgANDOR)

print("Sum AND: "+str(sumAND))            
print("Sum OR: "+str(sumOR))       
iouValue = (sumAND/sumOR)*100

print("IOU value %3.3f"%(iouValue))

























cv2.waitKey(0)
cv2.destroyAllWindows()

