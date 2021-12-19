## Right side detection

import cv2
import os
import numpy as np
import matplotlib.pyplot as plt  



def isBorder(img,iCent,jCent,fSz):
	fSz2 = fSz//2
	sumPix = float(0)
	for i in range(-fSz2,(fSz2+1),1):
		for j in range(-fSz2,(fSz2+1),1):
			sumPix = sumPix + float(img[iCent+i, jCent+j][0])

	bordVal = float(sumPix/(fSz*fSz))
	bordStat = 0
	if (bordVal > 0) and (bordVal<127):
		bordStat = 1
	
	return bordStat


def makeDot(img,i,j,dotOpt):
	if dotOpt==1:
		bordLineColor = [0,0,255]
		filledColor   = [255,100,255]
	if dotOpt==2:
		bordLineColor = [255,10,10]
		filledColor   = [255,255,0]
	
	delta = 11
	deltaMin1 = -(delta-1)
	for k in range(deltaMin1,delta,1):
		for l in range(deltaMin1,delta,1):
			img[(i+k),(j+l),0:3] = bordLineColor[0:3]

	delta = 7
	deltaMin1 = -(delta-1)
	for k in range(deltaMin1,delta,1):
		for l in range(deltaMin1,delta,1):
			img[(i+k),(j+l),0:3] = filledColor[0:3]
	return img


def drawAPolyLine(iLeft, jLeft, img, linePost):
	y = np.array(iLeft, float)
	x = np.array(jLeft, float)

	fit = np.polyfit(x, y, 2)

	a = fit[0]
	b = fit[1]
	c = fit[2]

	fit_equation = a * np.square(x) + b * x + c

	N = img.shape[1]

	xIn = []
	if linePost ==0:
		for jIdx in range(0, (N//2), 2):	
			xIn.append(float(jIdx))
		lineColor = (0, 255, 0)
	if linePost ==1:
		for jIdx in range((N//2),(N-1), 2):	
			xIn.append(float(jIdx))
		lineColor = (255, 0, 255)

	#print(xIn)
	#print(type(xIn))

	yFit =[]
	for xP in xIn:
		yVal = a*xP*xP + b*xP + c
		yFit.append(yVal)
	#print(yFit)
	#print(type(yFit))


	NFit = len(xIn)
	
	lineThickness = 3

	for i in range(NFit-2):
		start_point = (int(xIn[i]), int(yFit[i])) 
		end_point = (int(xIn[i+1]), int(yFit[i+1]))

		img = cv2.line(img, start_point, end_point, lineColor, lineThickness)
	
	return img
import random


os.system("cls")

foldNo = [0, 1, 2, 3]
rgbFold = [3, 0, 1, 2]

ratio = 0.4
kFoldIdx = 0

path = r"C:\Users\Esa\Pictures\_DATASET\unetpusbin\result_fold"+str(foldNo[kFoldIdx])
rgbPath =r"C:\Users\Esa\Pictures\_DATASET\unetpusbin\topCamSegmented\kFoldFolders\D"+str(rgbFold[kFoldIdx])
files = os.listdir(path)
# idx = int(input("Image index? "))
idx = random.randint(0,len(files))
idx = 33


print("============= File name is %s "%(files[idx]))

folderName = files[idx][:-4]
rgbPath = os.path.join(rgbPath,folderName)
rgbPath = os.path.join(rgbPath,"images")
rgbPath = os.path.join(rgbPath,files[idx])
print("============= RGB File name is %s "%(rgbPath))


oriImg = cv2.imread(rgbPath)
oriImgPerserved = oriImg.copy()

overlayImage = oriImg

img = cv2.imread(os.path.join(path,files[idx]))
img2 = img

M = oriImg.shape[0]
N = oriImg.shape[1]

dim =(int(ratio*N),int(ratio*M))
oriImg = cv2.resize(oriImg, dim, interpolation = cv2.INTER_AREA)
oriImgPerserved = cv2.resize(oriImgPerserved, dim, interpolation = cv2.INTER_AREA)

img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
# cv2.imshow("Original RGB Image", oriImg)


img2 = cv2.resize(img2, dim, interpolation = cv2.INTER_AREA)
# cv2.imshow("BW", img2)

img2BW = cv2.Canny(img2, 100, 150)
kernel = np.ones((1,1), np.uint8)
img2BW = cv2.dilate(img2BW, kernel, iterations=1)

M = img2BW.shape[0]
N = img2BW.shape[1]

sumPix = 0
iVanPoint = 0
while sumPix<10:
	for j in range (N):
		sumPix = sumPix + img2BW[iVanPoint,j]
	iVanPoint = iVanPoint + 1
print(iVanPoint)

color2 = (255, 0, 255) # BGR format
thickness = 1
# img2 = cv2.line(img2, [0,iVanPoint], [N,iVanPoint], color2, thickness)


sum_j = 0
sumPix = 0
for j in range(N):
    if img2BW[(iVanPoint-1),j] > 0:
        sum_j = sum_j + j
        sumPix = sumPix + 1
jAvg = round(sum_j / sumPix)
color3 = (0, 255, 0) # BGR format




dJ = int(N*0.2)
dI = dJ
img2 = cv2.line(img2, [jAvg-dJ,iVanPoint], [jAvg+dJ,iVanPoint], color2, thickness)
img2 = cv2.line(img2, [jAvg,iVanPoint-dI], [jAvg,iVanPoint+dI], color3, thickness)

oriImg = cv2.line(oriImg, [jAvg-dJ,iVanPoint], [jAvg+dJ,iVanPoint], color2, thickness)
oriImg = cv2.line(oriImg, [jAvg,iVanPoint-dI], [jAvg,iVanPoint+dI], color3, thickness)



edgeImgToShow = np.zeros((M,N,3), dtype=np.uint8)
thickVanPoint = 5
for k in range (3):
	edgeImgToShow[:,:,k] = img2BW
edgeImgToShow = cv2.line(edgeImgToShow, [jAvg-dJ,iVanPoint], [jAvg+dJ,iVanPoint], color2, thickVanPoint)
edgeImgToShow = cv2.line(edgeImgToShow, [jAvg,iVanPoint-dI], [jAvg,iVanPoint+dI], color3, thickVanPoint)
# cv2.imshow("Vanishing point detection", edgeImgToShow)


M = img2.shape[0]
N = img2.shape[1]

print("M %d"%(M))
print("N %d"%(N))

iRgEdge = iVanPoint
print("iRgEdge %d"%(iRgEdge))

sumPix = 0
while sumPix<100:
	sumPix=0
	for k in range (N-20,N,1):
		sumPix = sumPix + img2BW[iRgEdge,k]
	iRgEdge = iRgEdge + 1

print("iEgEdge: %d  sumPix: %d "%(iRgEdge,sumPix))
print("Final iRgEdge %d"%(iRgEdge))

iLfEdge = iVanPoint
sumPix = 0
while sumPix<100:
	sumPix=0
	for k in range (0,20,1):
		sumPix = sumPix + img2BW[iLfEdge,k]
	iLfEdge = iLfEdge + 1

iNrPoint = min(iLfEdge, iRgEdge)


rgSideI = []
rgSideJ = []
for i in range(iVanPoint, iRgEdge,10):
    sum_i = 0
    sum_j = 0
    nPix = 0
    for j in range(jAvg,N,1):
        if img2BW[i,j] == 255:
            rgSideI.append(i)
            rgSideJ.append(j)
            for k in range(-3,3,1):
                for l in range(-3,3,1):
                    img2[i+k,j+l,0] = 0
                    img2[i+k,j+l,1] = 0
                    img2[i+k,j+l,2] = 255
    
                    oriImg[i+k,j+l,0] = 0
                    oriImg[i+k,j+l,1] = 0
                    oriImg[i+k,j+l,2] = 255

                    edgeImgToShow[i+k,j+l,0] = 0
                    edgeImgToShow[i+k,j+l,1] = 0
                    edgeImgToShow[i+k,j+l,2] = 255

lfSideI = []
lfSideJ = []
for i in range(iVanPoint, iLfEdge,10):
    sum_i = 0
    sum_j = 0
    nPix = 0
    for j in range(0,jAvg,1):
        if img2BW[i,j] == 255:
            lfSideI.append(i)
            lfSideJ.append(j)
            for k in range(-3,3,1):
                for l in range(-3,3,1):
                    img2[i+k,j+l,0] = 240
                    img2[i+k,j+l,1] = 220
                    img2[i+k,j+l,2] = 60
    
                    oriImg[i+k,j+l,0] = 240
                    oriImg[i+k,j+l,1] = 220
                    oriImg[i+k,j+l,2] = 60

                    edgeImgToShow[i+k,j+l,0] = 240
                    edgeImgToShow[i+k,j+l,1] = 220
                    edgeImgToShow[i+k,j+l,2] = 60




print("------------")
print(len(rgSideI))
print(len(rgSideJ))

print(len(lfSideI))
print(len(lfSideJ))


nRight = len(rgSideJ)
nLeft  = len(lfSideJ)
nCenter = min(nRight, nLeft)


for idx in range (nCenter):
	iMid = round((rgSideI[idx] + lfSideI[idx])/2)
	jMid = round((rgSideJ[idx] + lfSideJ[idx])/2)
	for k in range(-3,3,1):
		for l in range(-3,3,1):
			img2[iMid+k,jMid+l,0] = 80
			img2[iMid+k,jMid+l,1] = 80
			img2[iMid+k,jMid+l,2] = 80
			
			oriImg[iMid+k,jMid+l,0] = 80
			oriImg[iMid+k,jMid+l,1] = 80
			oriImg[iMid+k,jMid+l,2] = 80

			edgeImgToShow[iMid+k,jMid+l,0] = 0
			edgeImgToShow[iMid+k,jMid+l,1] = 255
			edgeImgToShow[iMid+k,jMid+l,2] = 255




# cv2.imshow("Road Side Detection", edgeImgToShow)
# cv2.imshow("ORI RGB", oriImg)
# cv2.imshow("Img BW Edge", img2BW)
# cv2.imshow("Original Image with Line", img2)

print("Image index %d "%(idx))

rowImg1 = np.hstack((oriImgPerserved, img))
rowImg2 = np.hstack((edgeImgToShow, oriImg))
finalImage = np.vstack((rowImg1, rowImg2))
cv2.imshow("Final Image No:"+str(idx),finalImage)

cv2.imshow("BW", edgeImgToShow)

cv2.waitKey(0)
cv2.destroyAllWindows()
