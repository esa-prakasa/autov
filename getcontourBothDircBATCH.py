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
path = r"C:\Users\Esa\Pictures\_DATASET\unetpusbin\result_fold1"
savePath = r"C:\Users\Esa\Pictures\_DATASET\unetpusbin\result_fold1_edge"
files = os.listdir(path)

numOfFilesInFolder = len(files)

for idx in range(500,numOfFilesInFolder,1):
#for idx in range(3):

    img = cv2.imread(os.path.join(path,files[idx]))
    img2 = img

    M = img.shape[0]
    N = img.shape[1]

    iR = 0
    sumRow = 0
    lastSumRow = sumRow
    while lastSumRow <= 0:
        for j in range(N):
            sumRow = sumRow + float(img[iR,j,0])
        iR = iR +1
        #print("%d  %4.2f"%(iR, sumRow))
        lastSumRow = sumRow
        sumRow = 0
	
    Mmin = iR - 1


    #print("%d %d"%(M,N))
    #print("--------------------------")

    start_point = (0, 0) 
    end_point = (N, M)
    color = (0, 255, 0)
    thickness = 1
    #img = cv2.line(img, start_point, end_point, color, thickness)

    img2 = img.copy()


    NoOfDivision = 30
    rowDivSpace = M//NoOfDivision
    #print("NoOfDivision %d rowDivSpace %d "%(NoOfDivision,rowDivSpace))

    iLeft =[]
    jLeft =[]

    iRight =[]
    jRight =[]

    thisRow = 0

    bordRatioTh = 0.5
    errThr = 0.3

    color2 = (0, 0, 255) # BGR format

    #cv2.imshow("Original Image", img)

    Mmax = int(0.7*M)


    for i in range(Mmin,Mmax,rowDivSpace):
        start_point = (0,i)
        end_point = (N,i)
        img2 = cv2.line(img2, start_point, end_point, color2, thickness)
        



    for i in range(Mmin,Mmax,rowDivSpace):
        thisRow = 0
        for j in range(0,(int(0.8*N)),1):
            bordStat = isBorder(img,i,j,5)
            
            if (bordStat==1) and (thisRow==0) and (j>20) and (j<(N//2)):
                
                img2 = makeDot(img2, i, j,1)
                iLeft.append(i)
                jLeft.append(j)
                thisRow = 1



    for i in range(Mmin,Mmax,rowDivSpace):
        thisRow = 0
        for j in range(0,(int(0.8*N)),1):
            j2 = N - 10 - j
            bordStat = isBorder(img,i,j2,5)
            
            if (bordStat==1) and (thisRow==0) and (j2>(N//2)) and (j2<N):
                
                img2 = makeDot(img2, i, j2,2)
                iLeft.append(i)
                jLeft.append(j2)
                thisRow = 1





    #print("--------------------------")
    #N = len(iLeft)
    #print(N)
    #for i in range(N):
    #    print("i: %d  j: %d "%(iLeft[i], jLeft[i]))


    #img2 = drawAPolyLine(iLeft, jLeft,img2,0)
    #cv2.imshow("["+str(idx)+"] --- (2)"+files[idx], img2)

    cv2.imwrite(os.path.join(savePath,files[idx]), img2)
    print("%d of %d : File %s has been saved"%(idx,numOfFilesInFolder ,files[idx]))


    #print("Done! getcontourBothDirc3.py")

cv2.waitKey(0)
cv2.destroyAllWindows()
