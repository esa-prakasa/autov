import os
import cv2
import numpy as np
os.system("cls")

kfold = "_fold4"
NforDataset = 8000
rootPath = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\Purbaleunyi_Highway_1\\ds\\"

#srcFolderB  = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\dataset\\B\\"
#srcFolderNB = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\dataset\\NB\\"

#trainFolder = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\dataset\\"+kfold+"\\train\\"
#validFolder = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\dataset\\"+kfold+"\\valid\\"
#testFolder  = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\dataset\\"+kfold+"\\test\\"


srcFolderB  = rootPath+"_B\\"
srcFolderNB = rootPath+"_NB\\"

trainFolder = rootPath+kfold+"\\train\\"
validFolder = rootPath+kfold+"\\valid\\"
testFolder  = rootPath+kfold+"\\test\\"


print(trainFolder)

def getListIdxAll(files,pc1,pc2):
	N = len(files)
	b1 = int(N*pc1)
	train = files[:b1] 
	train2 = files[b1:]
	N = len(train2)
	b2 = int(N*(pc2/(1-pc1)))
	valid = train2[:b2] 
	test = train2[b2:] 
	return train, valid, test


def getListIdxPartial(files,N,pc1,pc2):
	b1 = int(N*pc1)
	train = files[:b1] 
	train2 = files[b1:N]
	N2 = len(train2)
	b2 = int(N2*(pc2/(1-pc1)))
	valid = train2[:b2] 
	test = train2[b2:] 
	return train, valid, test



###================= Prepare B data
itmIdx = 0
srcFiles = os.listdir(srcFolderB)

files = np.random.permutation(len(srcFiles))
#[train, valid, test] = getListIdxAll(files, 0.7, 0.2)

#nt = len(train)
#nv = len(valid)
#ns = len(test)
#tn = nt+nv+ns

#print("N train "+str(nt)+"  %2.3f "%(nt/tn))
#print("N valid "+str(nv)+"  %2.3f "%(nv/tn))
#print("N test "+str(ns)+"  %2.3f "%(ns/tn))

#print("--------------------")

[train, valid, test] = getListIdxPartial(files, NforDataset, 0.7, 0.2)

'''
nt = len(train)
nv = len(valid)
ns = len(test)
tn = nt+nv+ns
print("Total: "+str(tn))

print("N train "+str(nt)+"  %2.3f "%(nt/tn))
for i in range(5):
	print(train[i])

print("N valid "+str(nv)+"  %2.3f "%(nv/tn))
for i in range(5):
	print(valid[i])

print("N test "+str(ns)+"  %2.3f "%(ns/tn))
for i in range(5):
	print(test[i])
'''

Ntr = len(train)
for i in range(Ntr):
	idx = train[i]
	img = cv2.imread(srcFolderB+srcFiles[idx])
	cv2.imwrite((trainFolder+"_B\\"+srcFiles[idx]),img)
	#print(str(itmIdx)+"--"+trainFolder+"_B\\"+srcFiles[idx])
	print(str(itmIdx)+"--"+srcFiles[idx])
	itmIdx = itmIdx + 1


Nvl = len(valid)
for i in range(Nvl):
	idx = valid[i]
	img = cv2.imread(srcFolderB+srcFiles[idx])
	cv2.imwrite((validFolder+"_B\\"+srcFiles[idx]),img)
	print(str(itmIdx)+"--"+srcFiles[idx])
	itmIdx = itmIdx + 1

Nts = len(test)
for i in range(Nts):
	idx = test[i]
	img = cv2.imread(srcFolderB+srcFiles[idx])
	cv2.imwrite((testFolder+"_B\\"+srcFiles[idx]),img)
	print(str(itmIdx)+"--"+srcFiles[idx])
	itmIdx = itmIdx + 1



###================= Prepare NB data
itmIdx = 0
srcFiles = os.listdir(srcFolderNB)

files = np.random.permutation(len(srcFiles))
[train, valid, test] = getListIdxPartial(files, NforDataset, 0.7, 0.2)

Ntr = len(train)
for i in range(Ntr):
	idx = train[i]
	img = cv2.imread(srcFolderNB+srcFiles[idx])
	cv2.imwrite((trainFolder+"_NB\\"+srcFiles[idx]),img)
	print(str(itmIdx)+"--"+srcFiles[idx])
	itmIdx = itmIdx + 1

Nvl = len(valid)
for i in range(Nvl):
	idx = valid[i]
	img = cv2.imread(srcFolderNB+srcFiles[idx])
	cv2.imwrite((validFolder+"_NB\\"+srcFiles[idx]),img)
	print(str(itmIdx)+"--"+srcFiles[idx])
	itmIdx = itmIdx + 1

Nts = len(test)
for i in range(Nts):
	idx = test[i]
	img = cv2.imread(srcFolderNB+srcFiles[idx])
	cv2.imwrite((testFolder+"_NB\\"+srcFiles[idx]),img)
	print(str(itmIdx)+"--"+srcFiles[idx])
	itmIdx = itmIdx + 1


print("The data has been saved")