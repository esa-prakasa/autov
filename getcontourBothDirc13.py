## Right side detection

from ntpath import join
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt  


def createOverlayImage(rgbImage, bwImage):
	M = rgbImage.shape[0]
	N = rgbImage.shape[1]

	maskedImage = np.zeros((M,N,3), dtype=np.uint8)
	maskedImage = rgbImage

	for i in range(M):
		for j in range(N):
			if bwImage[i,j,0]>100:
				maskedImage[i,j,0] = 100 #B
				maskedImage[i,j,1] = rgbImage[i,j,2] #G
				maskedImage[i,j,2] = 100 #R

	return maskedImage



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
		lineColor = (0, 255, 255)
	if linePost ==1:
		for jIdx in range((N//2),(N-1), 2):	
			xIn.append(float(jIdx))
		lineColor = (0, 255, 255)

	#print(xIn)
	#print(type(xIn))

	yFit =[]
	for xP in xIn:
		yVal = a*xP*xP + b*xP + c
		yFit.append(yVal)
	#print(yFit)
	#print(type(yFit))


	NFit = len(xIn)
	
	lineThickness = 1

	for i in range(NFit-2):
		start_point = (int(xIn[i]), int(yFit[i])) 
		end_point = (int(xIn[i+1]), int(yFit[i+1]))

		img = cv2.line(img, start_point, end_point, lineColor, lineThickness)
	
	return img






import random

# os.system("cls")


foldNo = [0, 1, 2, 3]
rgbFold = [3, 0, 1, 2]

ratio = 0.4
kFoldIdx = 0

path = r"C:\Users\Esa\Pictures\_DATASET\unetpusbin\result_fold"+str(foldNo[kFoldIdx])
rgbPath =r"C:\Users\Esa\Pictures\_DATASET\unetpusbin\topCamSegmented\kFoldFolders\D"+str(rgbFold[kFoldIdx])
files = os.listdir(path)
# idx = int(input("Image index? "))
idx = random.randint(0,len(files))
# idx = 578  ## centered vanishing point
# idx = 33  ## centered vanishing point
# idx = 100  ## centered vanishing point
# idx = 500  ## vanishing point goes to left side
# idx = 700
# 
#   ## vanishing point goes to left side
# idx = 900  ## vanishing point goes to left side

# import random
# idx = random.randint(1, 900)


print("============= File name is %s "%(files[idx]))

folderName = files[idx][:-4]
rgbPath = os.path.join(rgbPath,folderName)
rgbPath = os.path.join(rgbPath,"images")
rgbPath = os.path.join(rgbPath,files[idx])
print("============= RGB File name is %s "%(rgbPath))

labelPath = r"C:\Users\Esa\Pictures\_DATASET\unetpusbin\topCamSegmented\kFoldFolders\D"+str(3)
labelPath = os.path.join(labelPath,folderName,"mask")
# labelPath = os.path.join(labelPath,files[idx])
# C:\Users\Esa\Pictures\_DATASET\unetpusbin\topCamSegmented\kFoldFolders\D0\00008_0ZxQhvAQvI\mask

labelPath = os.path.join(labelPath,files[idx])
print("FILENAME is >>>>> "+labelPath)

labelImg = cv2.imread(labelPath)
# img2 = img

M = labelImg.shape[0]
N = labelImg.shape[1]

dim =(int(ratio*labelImg.shape[1]),int(ratio*labelImg.shape[0]))
labelImg = cv2.resize(labelImg, dim, interpolation = cv2.INTER_AREA)











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



cv2.imshow("BW", img2)
img2Perserved = img2.copy()



# =======================
# Find the bottom border
# =======================

# M = img2.shape[0]
# N = img2.shape[1]

# i = M-5
# horzSum = 0
# for j in range(N):
# 	horzSum = horzSum + img2[i,j,0]
# print("Horizontal Sum")
# print(horzSum)
# print("Horizontal Sum")




img2BW = cv2.Canny(img2, 100, 150)
kernel = np.ones((1,1), np.uint8)
img2BW = cv2.dilate(img2BW, kernel, iterations=1)

edPathSave = r"C:\Users\Esa\Pictures\_DATASET\unetpusbin\edges"
edFileName = "edge_"+str(idx)+".png"
edFullPath = os.path.join(edPathSave, edFileName)
cv2.imwrite(edFullPath, img2BW)
cv2.imshow("img2BW edges", img2BW)



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



color2 = (255, 255, 255) # BGR format
color3 = (255, 255, 255) # BGR format

edgeImgToShow = np.zeros((M,N,3), dtype=np.uint8)
thickVanPoint = 3
for k in range (3):
	edgeImgToShow[:,:,k] = img2BW
edgeImgToShow = cv2.line(edgeImgToShow, [jAvg-dJ,iVanPoint], [jAvg+dJ,iVanPoint], color2, thickVanPoint)
edgeImgToShow = cv2.line(edgeImgToShow, [jAvg,iVanPoint-dI], [jAvg,iVanPoint+dI], color3, thickVanPoint)
# cv2.imshow("Vanishing point detection", edgeImgToShow)


M = img2.shape[0]
N = img2.shape[1]

print("M %d"%(M))
print("N %d"%(N))


stepSz = 3

rgSideI = []
rgSideJ = []
lfSideI = []
lfSideJ = []
mid_I = []
mid_J = []

Ndiv2 = N//2

print("iVanPoint: %d "%(iVanPoint))

M = img2BW.shape[0]


for i in range(iVanPoint, int(0.9*M), stepSz):
    j1 = 0
    while j1 < jAvg:
        if img2BW[i,j1] > 200:
            lfSideI.append(i)
            lfSideJ.append(j1)
        j1 = j1 + 1

    j2 = Ndiv2
    while (j2 >= jAvg
	
	) and (j2<N):
        if img2BW[i,j2] > 200:
            rgSideI.append(i)
            rgSideJ.append(j2)
        j2 = j2 + 1

for x in lfSideI:
    print("Left: %d "%(x))

print(lfSideI)
print(N)

print("Helooooo")




NLeft = len(lfSideI)
for i in range(NLeft):
    for k in range(-2,2,1):
        for l in range(-2,2,1):
            oriImg[lfSideI[i]+k, lfSideJ[i]+l, 0] = 255
            oriImg[lfSideI[i]+k, lfSideJ[i]+l, 1] = 0
            oriImg[lfSideI[i]+k, lfSideJ[i]+l, 2] = 0


NRight = len(rgSideI)
for i in range(NRight):
    for k in range(-2,2,1):
        for l in range(-2,2,1):
            oriImg[rgSideI[i]+k, rgSideJ[i]+l, 0] = 0
            oriImg[rgSideI[i]+k, rgSideJ[i]+l, 1] = 0
            oriImg[rgSideI[i]+k, rgSideJ[i]+l, 2] = 255



rowImg1 = np.hstack((oriImgPerserved, img))
rowImg2 = np.hstack((edgeImgToShow, oriImg))
finalImage = np.vstack((rowImg1, rowImg2))
cv2.imshow("Final Image No:"+str(idx),finalImage)



# pictFigure = np.hstack((oriImgPerserved, img2Perserved))

oriImgPerserved2 = oriImgPerserved.copy()
overlayImg = createOverlayImage(oriImgPerserved2, img2Perserved)

pictFigure1 = np.hstack((oriImgPerserved, labelImg))
pictFigure2 = np.hstack((img2Perserved, overlayImg))
pictFigure  = np.vstack((pictFigure1, pictFigure2))
cv2.imshow("Manuscript Figure "+str(idx),pictFigure)

os.system("cls")
print("============= Helooooo ===============")
print(files[idx]+" ("+str(idx)+")")

cv2.waitKey(0)
cv2.destroyAllWindows()
