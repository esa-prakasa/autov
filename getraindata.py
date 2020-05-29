import numpy as np
import cv2
import os

os.system("cls")

usedFileNo = int(input("Input usedFileNo (between 0 and 4) "))

labPath = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\label\\"
labFiles = os.listdir(labPath)
for i in range(len(labFiles)):
	print(labFiles[i])

labImg = cv2.imread(labPath+labFiles[usedFileNo])



rgbPath   = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\rgbs\\"
rgbFiles = os.listdir(rgbPath)
for i in range(len(rgbFiles)):
	print(rgbFiles[i])

rgbImg = cv2.imread(rgbPath+rgbFiles[usedFileNo])


M = labImg.shape[0]
N = labImg.shape[1]




print(M)
print(N)

fSz = 20

#subImg = np.zeros((100,100,3), dtype="uint8")
#subImg = labImg[0:int(0+100), 0:int(0+100), :]

#print(subImg.shape[0])
#print(subImg.shape[1])
#cv2.imshow("subImg", subImg)


def compArea(subImgLab):
	M = subImgLab.shape[0]
	N = subImgLab.shape[1]
	area = 0

	for i in range(M):
		for j in range(N):
			if (subImgLab[i,j,0]==255):
				area = area + 1

	pctArea = area/(M*N)

	return pctArea, area 





idx = 0
for i in range(0,(M-fSz),fSz):
	for j in range(0,(N-fSz),fSz):
		
		print("%d %d"%(i,j))
		ic = int(i)
		jc = int(j)
		subImgLab = labImg[ic:(ic+fSz), jc:(jc+fSz), :]
		subImgRGB = rgbImg[ic:(ic+fSz), jc:(jc+fSz), :]

		[pctArea, area] = compArea(subImgLab)
		print("%d   %3.2f  "%(idx, pctArea))

		pctAreaS =("%2.2f"%(pctArea))

		if ((pctArea>=0.25) and (pctArea<=0.85)):
			cv2.imwrite("C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\temp\\B_lab\\"+str(usedFileNo)+"__"+str(ic)+"_"+str(jc)+"_"+pctAreaS+".jpg", subImgLab)
			cv2.imwrite("C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\temp\\B\\"+str(usedFileNo)+"__"+str(ic)+"_"+str(jc)+"_"+pctAreaS+".jpg", subImgRGB)
		if ((pctArea<0.25) or (pctArea>0.85)):
			cv2.imwrite("C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\temp\\NB_lab\\"+str(usedFileNo)+"__"+str(ic)+"_"+str(jc)+"_"+pctAreaS+".jpg", subImgLab)
			cv2.imwrite("C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\temp\\NB\\"+str(usedFileNo)+"__"+str(ic)+"_"+str(jc)+"_"+pctAreaS+".jpg", subImgRGB)
		idx = idx + 1

	
print(labFiles[0])
print(rgbFiles[0])

cv2.waitKey(0)
cv2.destroyAllWindows()

