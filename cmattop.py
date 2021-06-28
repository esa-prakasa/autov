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

'''
for i in range(10):

	fullPath0 = rootPath+"\\"+folderNames[i]+"\\masks\\_mask.png" 
	fullPath1 = resPath+"\\"+resNames[i]
	img0 = cv2.imread(fullPath0)
	M = img0.shape[0]
	N = img0.shape[1]
	img1 = cv2.imread(fullPath1)

	img1 = cv2.resize(img1, (N,M), interpolation = cv2.INTER_AREA)


	imgF = np.hstack((img0,img1))

	#cv2.imshow(str(i),imgF)


	print(fullPath0)
	print(fullPath1)

'''
idx= int(input("What is the index? "))

#fullPath0 = rootPath+"\\"+folderNames[idx]+"\\masks\\_mask.png" 
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



## colouring parts
imgPath = rootPath+"\\"+folderNames[idx]+"\\images\\" #rgb images png"
imgFullPath = imgPath+"\\"+folderNames[idx]+".png"
imgRGB = cv2.imread(imgFullPath)
imgRGB = cv2.resize(imgRGB, (N,M), interpolation = cv2.INTER_AREA)

labImgRGB = imgRGB # np.zeros((M,N,3), dtype = np.uint8)
for i in range(M):
	for j in range(N):
		if img0[i,j,0] == 255:
			labImgRGB[i,j,0] = 0



cv2.imshow("RGB image", labImgRGB)











actual = []
predicted = []

for i in range(M):
	for j in range(N):
		if img0[i,j,0] >= 200:
			val = 1
		if img0[i,j,0] < 200:
			val = 0
		actual.append(val)
		if img1[i,j,0] >= 200:
			val = 1
		if img1[i,j,0] < 200:
			val = 0
		predicted.append(val)

print(" Image "+str(idx))
# confusion matrix
matrix = confusion_matrix(actual,predicted, labels=[1,0])
print('Confusion matrix : \n',matrix)

# outcome values order in sklearn
tp, fn, fp, tn = confusion_matrix(actual,predicted,labels=[1,0]).reshape(-1)
# print('Outcome values : \n', tp, fn, fp, tn)

# classification report for precision, recall f1-score and accuracy
matrix = classification_report(actual,predicted,labels=[1,0])
print('Classification report : \n',matrix)
















cv2.waitKey(0)
cv2.destroyAllWindows()

