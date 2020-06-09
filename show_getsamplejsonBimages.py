#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#XXXXXXXXXXXXXXXXXXXX BODER EXTRACTION -- NON BODER EXTRACTION XXXXXXXXXXXXXXXXXXXXXXX
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#XXXXXXXXXXXXXXXXXXXX BODER EXTRACTION -- NON BODER EXTRACTION XXXXXXXXXXXXXXXXXXXXXXX
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

f1 = 40
f2 = f1+1
for fileIdx in range (f1,f2,1): #(NAnnFiles):

	# Opening JSON file 
	f = open(annPath+annFiles[fileIdx])
	img = cv2.imread(imgPath+imgFiles[fileIdx]) 
	img2 = img.copy()

	blackImg = np.zeros((img.shape[0],img.shape[1],3), dtype =np.uint8)

	data = json.load(f) 
 
	color = (0, 255, 0)
	whiteCol = (255, 255, 255)
	thickness = 3
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
	#cv2.imshow(str(fileIdx)+"img", img) 
	#cv2.imshow(str(fileIdx)+"anno", blackImg) 

	mergeImg = cv2.vconcat((img,blackImg))
	cv2.imshow("merge", mergeImg)

	pathToSaveM = "C:\\Users\\INKOM06\\Documents\\[0--KEGIATAN-Ku-2020\\2020.01-006-Autonomous Vehicle Project\\borderDetection202005\\presentation01\\"
	pathToSaveM = pathToSaveM+"merg"+str(fileIdx)+".png"
	cv2.imwrite(pathToSaveM, mergeImg)




	#bwPath = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\Purbaleunyi_Highway_1\\ds\\"

	#cv2.imwrite(bwPath+"_bw.jpg", blackImg)
	#blackImg2 = cv2.imread(bwPath+"_bw.jpg")

	#cv2.imshow("blackImg2 ", blackImg2)


	def getSubImgRGB(img, fSz, ic, jc):
		f2 = fSz //2
		subImgRGB = np.zeros([fSz,fSz,3], dtype = np.uint8)
		subImgRGB = img[(ic-f2):(ic+f2), (jc-f2):(jc+f2), :]

		return subImgRGB


	'''
	fSz = 30

	print("------------------ pt -----------------")
	pt = getLineCoordinates(blackImg2)
	NPt = len(pt)
	print(NPt)
	for i in range(NPt):
		print(str(i)+" "+str(pt[i]))
		subImgRGB = getSubImgRGB(img2, fSz, pt[i][0], pt[i][1])
		#cv2.imshow(str(i), subImgRGB)
		idxStr = str(1000+i)
		idxStr = idxStr[1:]
		fileName = imgFiles[fileIdx]
		fileName = fileName[:-4] 
		fileName = fileName+idxStr+"_B.jpg"
		totalSubImgFiles = totalSubImgFiles + 1
		totalFileSize = totalSubImgFiles*(0.002)

		print(str(totalSubImgFiles)+" "+str(i)+" "+fileName+ " ===> %5.2f"%(totalFileSize))
		#cv2.imwrite(pathToSave_B+fileName, subImgRGB)

'''


cv2.waitKey(0)
cv2.destroyAllWindows()
