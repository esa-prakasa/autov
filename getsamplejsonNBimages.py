#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#XXXXXXXXXXXXXXXXXXXX NON BODER EXTRACTION -- NON BODER EXTRACTION XXXXXXXXXXXXXXXXXXX
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#XXXXXXXXXXXXXXXXXXXX NON BODER EXTRACTION -- NON BODER EXTRACTION XXXXXXXXXXXXXXXXXXX
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX



import os  
import json 
import cv2
import numpy as np
import random as rd

os.system("cls")

def getLineCoordinates(blackImg):
	M = blackImg.shape[0]
	N = blackImg.shape[1]
	pt = []
	idx = 0
	for i in range(M):
		for j in range(N):
			if (blackImg[i,j,0] == 255):
				pt.append([i,j])
				#print(pt[idx])
				idx = idx + 1
	return pt


annPath = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\Purbaleunyi_Highway_1\\ds\\ann\\"
imgPath = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\Purbaleunyi_Highway_1\\ds\\img\\"

pathToSave_B  = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\Purbaleunyi_Highway_1\\ds\\_B\\"
pathToSave_NB = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\Purbaleunyi_Highway_1\\ds\\_NB\\" 

annFiles = os.listdir(annPath)
imgFiles = os.listdir(imgPath)

NAnnFiles = (len(annFiles))

#for i in range(N):
#	print(annFiles[i]+" --- "+imgFiles[i])
#fileIdx = int(input("Give file index (max 40) "))

totalSubImgFiles = 0


#for fileIdx in range (1): #(NAnnFiles):
for fileIdx in range (NAnnFiles):

	# Opening JSON file 
	f = open(annPath+annFiles[fileIdx])
	img = cv2.imread(imgPath+imgFiles[fileIdx]) 
	rgbImg = img.copy()

	blackImg = np.zeros((img.shape[0],img.shape[1]), dtype =np.uint8)

	data = json.load(f) 
 
	color = (0, 255, 0)
	whiteCol = (255, 255, 255)
	thickness = 8
	whiteThickness = 2

	NPoints = (len(data['objects']))   # number of lines 

	for i in range(NPoints):
		lineData = data['objects'][i]['points']['exterior']
		print(data['objects'][i]['points']['exterior'])
		N = (len(lineData))
		strIdx = 0
		for j in range(1,N):
			print(lineData[j])
			endIdx = j
			x  =  lineData[strIdx][0]
			y   = lineData[strIdx][1]
			start_point = (x,y)

			x  =  lineData[endIdx][0]
			y   = lineData[endIdx][1]
			end_point = (x, y) #lineData[1]

			img = cv2.line(img, start_point, end_point, color, thickness)
			blackImg = cv2.line(blackImg, start_point, end_point, whiteCol, whiteThickness)

			strIdx = endIdx
 
	#cv2.imshow("img", img) 
	f.close() 
	#cv2.imshow("anno", blackImg) 


	bwPath = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\Purbaleunyi_Highway_1\\ds\\"

	cv2.imwrite(bwPath+"_bw.jpg", blackImg)
	blackImg2 = cv2.imread(bwPath+"_bw.jpg")

	#cv2.imshow("blackImg2 ", blackImg2)


	def getSubImgRGB(img, fSz, ic, jc):
		f2 = fSz //2
		subImgRGB = np.zeros([fSz,fSz,3], dtype = np.uint8)
		subImgRGB = img[(ic-f2):(ic+f2-1), (jc-f2):(jc+f2-1), :]

		return subImgRGB


	fSz = 30


	M = rgbImg.shape[0]
	N = rgbImg.shape[1]



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
	dcP = 8
	for i in range((M//3),(M-fSz),dcP):
		for j in range((N//3),(N-fSz),dcP):
			#print("%d %d"%(i,j))
			ic = int(i)
			jc = int(j)

			subImgLab = blackImg2[ic:(ic+fSz), jc:(jc+fSz), :]
			subImgRGB = rgbImg[ic:(ic+fSz), jc:(jc+fSz), :]

			[pctArea, area] = compArea(subImgLab)
			#print("%d   %3.2f  "%(idx, pctArea))

			pctAreaS =("%2.2f"%(pctArea))

			if (pctArea==0):
				idxStr = str(1000+idx)
				idxStr = idxStr[1:]
				fileName = imgFiles[fileIdx]
				fileName = fileName[:-4]
				fileName = fileName+idxStr+"_NB.jpg"
				totalSubImgFiles = totalSubImgFiles + 1
				totalFileSize = totalSubImgFiles*(0.002)
			#print(str(totalSubImgFiles)+" "+str(i)+" "+fileName+ " ===> %5.2f"%(totalFileSize))
				print(str(totalSubImgFiles)+"    "+str(idx)+ " ===> %5.2f"%(totalFileSize))
				cv2.imwrite(pathToSave_NB+fileName, subImgRGB)
				idx = idx + 1



cv2.waitKey(0)
cv2.destroyAllWindows()
