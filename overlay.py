import cv2
import os
import numpy as np



os.system('cls')



rootPath = r'C:\Users\INKOM06\Pictures\roadDataset\pusbin1130segment\img'
resPath = r'C:\Users\INKOM06\Pictures\roadDataset\pusbin1130segment\result4\_hpcEpoch100'

rgbNames = os.listdir(rootPath)
resNames = os.listdir(resPath)

imgIdx = 0
resImg = cv2.imread(os.path.join(resPath,resNames[imgIdx]))
cv2.imshow("Res Img",resImg)

rgbImg = cv2.imread(os.path.join(rootPath,resNames[imgIdx]))
cv2.imshow("RGB Img", rgbImg)


M = resImg.shape[0]
N = resImg.shape[1]

for i in range(M):
	for j in range(N):
		


cv2.waitKey(0)
cv2.destroyAllWindows()
