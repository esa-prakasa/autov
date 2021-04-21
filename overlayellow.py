import numpy as np
import cv2
import os



path = r"C:\Users\INKOM06\Pictures\roadDataset\pusbin1130segment\img\00000_fbHpiTTp6m.png"

img = cv2.imread(path)
M = img.shape[0]
N = img.shape[1]
img2 = np.zeros((M,N,3), dtype=np.uint8)

for i in range(M):
	for j in range(N):
		img2[i,j,0] = img[i,j,0]
		img2[i,j,1] = img[i,j,1]
		img2[i,j,2] = img[i,j,2]

		if ((i >(M//2)) & (j>(N//2))):
			img2[i,j,0] = 0
			
			diff = 100

			addVal = img[i,j,1] + diff
			if addVal > 255:
				img2[i,j,1] = 255
			if addVal<=255:
				img2[i,j,1] = addVal

			addVal = img[i,j,2] + diff
			if addVal > 255:
				img2[i,j,2] = 255
			if addVal<=255:
				img2[i,j,2] = addVal



cv2.imshow("RGB", img2)


cv2.waitKey(0)
cv2.destroyAllWindows()
