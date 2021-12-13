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




os.system("cls")
#path = r"C:\Users\Esa\Pictures\_DATASET\unetpusbin\result_fold1"
path = r"C:\Users\Esa\Pictures\_DATASET\unetpusbin\result_fold2"
files = os.listdir(path)


import random


# idx = int(input("Image index? "))
idx = random.randint(0,300)
# idx = 135


print("============= File name is %s "%(files[idx]))

folderName = files[idx][:-4]
rgbPath =r"C:\Users\Esa\Pictures\_DATASET\unetpusbin\topCamSegmented\kFoldFolders\D1"
rgbPath = os.path.join(rgbPath,folderName)
rgbPath = os.path.join(rgbPath,"images")
rgbPath = os.path.join(rgbPath,files[idx])
print("============= RGB File name is %s "%(rgbPath))

oriImg = cv2.imread(rgbPath)
overlayImage = oriImg
cv2.imshow("ORI RGB", oriImg)

img = cv2.imread(os.path.join(path,files[idx]))
img2 = img

# M = img.shape[0]
# N = img.shape[1]

# for i in range(M):
#     for j in range(N):
#         if img2[i,j,0] > 126:
#             #overlayImage[i,j,0] = oriImg[i,j,0] + 10
#             overlayImage[i,j,1] = 0
#             #overlayImage[i,j,2] = oriImg[i,j,2] 
            
# cv2.imshow("Overlay Image", overlayImage)


cv2.imshow("'BW", img2)
img2BW = cv2.Canny(img2, 100, 150)
# cv2.imshow("Edge "+str(idx), img2BW)

M = img2BW.shape[0]
N = img2BW.shape[1]

iVanPoint = 0
sumPix = 0
while sumPix<10:
	for j in range (N):
		sumPix = sumPix + img2BW[iVanPoint,j]
	iVanPoint = iVanPoint + 1

print(iVanPoint)

color2 = (0, 0, 255) # BGR format
thickness =2
img2 = cv2.line(img2, [0,iVanPoint], [N,iVanPoint], color2, thickness)
# cv2.imshow("Original Image with Line", img)


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

delta = 10
start_point = (N-(2*delta),iRgEdge-delta)  
end_point = (N-1,(iRgEdge+delta)) 



color = (0, 0, 255)
thickness = 2
image = cv2.rectangle(img2, start_point, end_point, color, thickness)

cv2.imshow("Edge with lines"+str(idx), img2)











# iR = 0
# sumRow = 0
# lastSumRow = sumRow
# while lastSumRow <= 0:
# 	for j in range(N):
# 		sumRow = sumRow + float(img[iR,j,0])
# 	iR = iR +1
# 	print("%d  %4.2f"%(iR, sumRow))
# 	lastSumRow = sumRow
# 	sumRow = 0
	
# Mmin = iR - 1


# #print("%d %d"%(M,N))
# #print("--------------------------")

# start_point = (0, 0) 
# end_point = (N, M)
# color = (0, 255, 0)
# thickness = 1
# #img = cv2.line(img, start_point, end_point, color, thickness)

# img2 = img.copy()


# NoOfDivision = 30
# rowDivSpace = M//NoOfDivision
# print("NoOfDivision %d rowDivSpace %d "%(NoOfDivision,rowDivSpace))

# iLeft =[]
# jLeft =[]

# iRight =[]
# jRight =[]

# thisRow = 0

# bordRatioTh = 0.5
# errThr = 0.3

# color2 = (0, 0, 255) # BGR format

# #cv2.imshow("Original Image", img)

# Mmax = int(0.8*M)

# thickness =2
# for i in range(Mmin,Mmax,rowDivSpace):
# 	start_point = (0,i)
# 	end_point = (N,i)
# 	img2 = cv2.line(img2, start_point, end_point, color2, thickness)

# cv2.imshow("Grid", img2)


# #For detectiing the left edge
# for i in range(Mmin,Mmax,rowDivSpace):
# 	thisRow = 0
# 	for j in range(0,(int(0.8*N)),1):
# 		bordStat = isBorder(img,i,j,5)
        
# 		if (bordStat==1) and (thisRow==0) and (j>20) and (j<(N//2)):
			
# 			img2 = makeDot(img2, i, j,1)
# 			iLeft.append(i)
# 			jLeft.append(j)
# 			thisRow = 1


# x1 = np.asfarray(iLeft, dtype="int8")
# y1 = np.asfarray(jLeft, dtype="int8")

# fit = np.polyfit(x1, y1, 2)
# a = fit[0]
# b = fit[1]
# c = fit[2]

# print(a)
# print(b)
# print(c)

# Ny1 = y1.size
# y = np.arange(x1[0],x1[Ny1-1],1)
# x = a * np.square(y) + b * y + c

# xI = x.astype(int)  # j index
# yI = y.astype(int)  # i index

# # print(x)
# # print(" ----------------- ")
# # print(y)
# # print(" ----------------- ")
# # print(yI)
# # print("N %d "%(Npts))
# # print(xI[1])


# color3 = (0, 255, 0) # BGR format
# Npts = x.size
# for i in range(1,Npts,1):
# 	start_point = (xI[i-1],yI[i-1])
# 	end_point = (xI[i],yI[i])
# 	img2 = cv2.line(img2, start_point, end_point, color3, thickness)



# iLeft =[]
# jLeft =[]




# #### For detecting the right edge
# for i in range(Mmin,Mmax,rowDivSpace):
# 	thisRow = 0
# 	for j in range(0,(int(0.8*N)),1):
# 		j2 = N - 10 - j
# 		bordStat = isBorder(img,i,j2,5)
        
# 		if (bordStat==1) and (thisRow==0) and (j2>(N//2)) and (j2<N):
			
# 			img2 = makeDot(img2, i, j2,2)
# 			iLeft.append(i)
# 			jLeft.append(j2)
# 			thisRow = 1


# x1 = np.asfarray(iLeft, dtype="int8")
# y1 = np.asfarray(jLeft, dtype="int8")

# fit = np.polyfit(x1, y1, 2)
# a = fit[0]
# b = fit[1]
# c = fit[2]

# print(a)
# print(b)
# print(c)

# Ny1 = y1.size
# y = np.arange(x1[0],x1[Ny1-1],1)
# x = a * y**2 + b * y + c

# xI = x.astype(int)  # j index
# yI = y.astype(int)  # i index


# color3 = (0, 255, 0) # BGR format
# Npts = x.size
# for i in range(1,Npts,1):
# 	start_point = (xI[i-1],yI[i-1])
# 	end_point = (xI[i],yI[i])
# 	img2 = cv2.line(img2, start_point, end_point, color3, thickness)





# # #print("--------------------------")
# N = len(iLeft)
# print(N)
# for i in range(N):
# 	print("i: %d  j: %d "%(iLeft[i], jLeft[i]))





# cv2.imshow("["+str(idx)+"] --- (2)"+files[idx], img2)


# print("Done! getcontourBothDirc4.py")

cv2.waitKey(0)
cv2.destroyAllWindows()
