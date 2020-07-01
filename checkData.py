# Tihis file is used to check the original and labelled data

import cv2
import os
import matplotlib.pyplot as plt
import time

start_time = time.time()

os.system("cls")



path = "C:\\Users\\INKOM06\\Pictures\\roadlane-detection-evaluation-2013\\training\\"
#path = "C:\\Users\\INKOM06\\Pictures\\jagung2020\\largeDataSet\\true\\"
labelPath = path + "gt_image_2\\"
oriPath   = path + "image_2\\"

print(labelPath)
print(oriPath)


labFiles = []
oriFiles = []

#for idx in range(100):
# r=root, d=directories, f = files
#idx = 0
for r, d, f in os.walk(labelPath):
	for file in f:
		labFilesData = os.path.join(r, file)
		labFiles.append(labFilesData)
#		idx = idx + 1
#		print(str(idx)+":"+labFilesData)

nLab = (len(labFiles))

print("-----------------\n")
# r=root, d=directories, f = files
#idx = 0
for r, d, f in os.walk(oriPath):
	for file in f:
		oriFilesData = os.path.join(r, file)
		oriFiles.append(oriFilesData)
#		idx = idx + 1
#		print(str(idx)+":"+oriFilesData)

nOri = (len(oriFiles))

N = min(nLab, nOri)
NFin = N

for i in range(N):
	strOri = (str(i)+": "+oriFiles[i])
	strLab = (str(i)+": "+labFiles[i])
	oriFileName = (strOri[-13:])
	labFileName = (strLab[-18:])
	print(str(i)+": "+oriFileName)
	print(str(i)+": "+labFileName)
	if oriFileName[-10:] == labFileName[-10:]:
		print("====MATCH!!!====") 
	if oriFileName[-10:] != labFileName[-10:]:
		print("InEqual!==================================*****=========>>>") 
		NFin = NFin - 1
	print("-- -- -- ")


print("%d of %d are match "%(NFin,N))

savePath = path + "needtocheck\\"
savePath2 = path + "overlay\\"

ratio = 0.5

NofImg = min(NFin,N)

for idx in range(NofImg):

	oriImg = cv2.imread(oriFiles[idx])
	labImg = cv2.imread(labFiles[idx])


	M = labImg.shape[0]
	N = labImg.shape[1]

	labImg = cv2.resize(labImg, (int(ratio*N),int(ratio*M)))
	oriImg = cv2.resize(oriImg, (int(ratio*N),int(ratio*M)))

	combImg = oriImg.copy()

	M = labImg.shape[0]
	N = labImg.shape[1]

	oriImg2 = oriImg[(M//2):M,:,:]
	labImg2 = labImg[(M//2):M,:,:]
	olayImg = oriImg2.copy()

	mO = oriImg2.shape[0]
	nO = oriImg2.shape[1]

	for i in range(mO):
		for j in range(nO):
			for k in range(3):
				combImg[i,j,k] = oriImg2[i,j,k] 
	
	mRef = mO-1

	mO = labImg2.shape[0]
	nO = labImg2.shape[1]

	for i in range(mO):
		for j in range(nO):
			for k in range(3):
				combImg[(i+mRef),j,k] = labImg2[i,j,k] 

	strComb = (str(idx)+": "+oriFiles[idx])
	combFileName = (strComb[-13:])
	print(combFileName)
	#cv2.imshow(str(idx),combImg)

	### create overlay image
	mO = oriImg2.shape[0]
	nO = oriImg2.shape[1]
	for i in range(mO):
		for j in range(nO):
			if labImg2[i,j,0] == 255:
				olayImg[i,j,0] = 0



	#cv2.imshow(str(idx),olayImg)
	cv2.imwrite((savePath+"comb_"+combFileName+".png"),combImg)
	cv2.imwrite((savePath2+"olay_"+combFileName+".png"),olayImg)







deltaTime = time.time() - start_time 
print("--- %5.5s seconds ---" % (deltaTime))


cv2.waitKey()
cv2.destroyAllWindows()
