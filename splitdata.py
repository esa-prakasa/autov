import os
import cv2
import numpy as np
os.system("cls")


srcFolderB  = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\dataset\\B\\"
srcFolderNB = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\dataset\\NB\\"


trainFolder = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\dataset\\_fold3\\train\\"
validFolder = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\dataset\\_fold3\\valid\\"
testFolder  = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\dataset\\_fold3\\test\\"


def getListIdx(files,pc1,pc2):
	N = len(files)
	b1 = int(N*pc1)
	train = files[:b1] 
	train2 = files[b1:]
	N = len(train2)
	b2 = int(N*(pc2/(1-pc1)))
	valid = train2[:b2] 
	test = train2[b2:] 
	return train, valid, test



###================= Prepare B data
itmIdx = 0
srcFiles = os.listdir(srcFolderB)

files = np.random.permutation(len(srcFiles))
[train, valid, test] = getListIdx(files, 0.7, 0.2)

Ntr = len(train)
for i in range(Ntr):
	idx = train[i]
	img = cv2.imread(srcFolderB+srcFiles[idx])
	cv2.imwrite((trainFolder+"B\\"+srcFiles[idx]),img)
	print(str(itmIdx)+"--"+srcFiles[idx])
	itmIdx = itmIdx + 1

Nvl = len(valid)
for i in range(Nvl):
	idx = valid[i]
	img = cv2.imread(srcFolderB+srcFiles[idx])
	cv2.imwrite((validFolder+"B\\"+srcFiles[idx]),img)
	print(str(itmIdx)+"--"+srcFiles[idx])
	itmIdx = itmIdx + 1

Nts = len(test)
for i in range(Nts):
	idx = test[i]
	img = cv2.imread(srcFolderB+srcFiles[idx])
	cv2.imwrite((testFolder+"B\\"+srcFiles[idx]),img)
	print(str(itmIdx)+"--"+srcFiles[idx])
	itmIdx = itmIdx + 1



###================= Prepare NB data
itmIdx = 0
srcFiles = os.listdir(srcFolderNB)

files = np.random.permutation(len(srcFiles))
[train, valid, test] = getListIdx(files, 0.7, 0.2)

Ntr = len(train)
for i in range(Ntr):
	idx = train[i]
	img = cv2.imread(srcFolderNB+srcFiles[idx])
	cv2.imwrite((trainFolder+"NB\\"+srcFiles[idx]),img)
	print(str(itmIdx)+"--"+srcFiles[idx])
	itmIdx = itmIdx + 1

Nvl = len(valid)
for i in range(Nvl):
	idx = valid[i]
	img = cv2.imread(srcFolderNB+srcFiles[idx])
	cv2.imwrite((validFolder+"NB\\"+srcFiles[idx]),img)
	print(str(itmIdx)+"--"+srcFiles[idx])
	itmIdx = itmIdx + 1

Nts = len(test)
for i in range(Nts):
	idx = test[i]
	img = cv2.imread(srcFolderNB+srcFiles[idx])
	cv2.imwrite((testFolder+"NB\\"+srcFiles[idx]),img)
	print(str(itmIdx)+"--"+srcFiles[idx])
	itmIdx = itmIdx + 1



print("The data has been saved")