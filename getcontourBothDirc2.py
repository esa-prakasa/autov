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

	bordVal = sumPix/(fSz*fSz)

#	print(bordVal)
#	print("%d %d :  %3.3f"%(iCent,jCent,bordVal))
	bordStat = bordVal


	return bordStat


def makeDot(img,i,j):

	delta = 11
	deltaMin1 = -(delta-1)
	for k in range(deltaMin1,delta,1):
		for l in range(deltaMin1,delta,1):
			img[(i+k),(j+l),0] = 0
			img[(i+k),(j+l),1] = 0
			img[(i+k),(j+l),2] = 255


	delta = 7
	deltaMin1 = -(delta-1)
	for k in range(deltaMin1,delta,1):
		for l in range(deltaMin1,delta,1):
			img[(i+k),(j+l),0] = 0
			img[(i+k),(j+l),1] = 255
			img[(i+k),(j+l),2] = 255


	return img


def makeDot2(img,i,j):

	delta = 11
	deltaMin1 = -(delta-1)
	for k in range(deltaMin1,delta,1):
		for l in range(deltaMin1,delta,1):
			img[(i+k),(j+l),0] = 0
			img[(i+k),(j+l),1] = 0
			img[(i+k),(j+l),2] = 255


	delta = 7
	deltaMin1 = -(delta-1)
	for k in range(deltaMin1,delta,1):
		for l in range(deltaMin1,delta,1):
			img[(i+k),(j+l),0] = 0
			img[(i+k),(j+l),1] = 255
			img[(i+k),(j+l),2] = 0


	return img




def drawAPolyLine(iLeft, jLeft, img, linePost):
	y = np.array(iLeft, float)
	x = np.array(jLeft, float)

#	fit = np.polyfit(x, y, 2)
	fit = np.polyfit(x, y, 3)
	a = fit[0]
	b = fit[1]
	c = fit[2]
	d = fit[3]
#	fit_equation = a * np.square(x) + b * x + c
	fit_equation = a * np.square(x)*x + b * np.square(x) + c*x + d

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
		yVal = a*xP*xP*xP + b*xP*xP + c*xP + d
		yFit.append(yVal)
	#print(yFit)
	#print(type(yFit))


	NFit = len(xIn)
	
	lineThickness = 3

	for i in range(NFit-2):
		#print("i  %d"%(i))
		start_point = (int(xIn[i]), int(yFit[i])) 
		end_point = (int(xIn[i+1]), int(yFit[i+1]))
		img = cv2.line(img, start_point, end_point, lineColor, lineThickness)
	
	return img




os.system("cls")
path = r"C:\Users\Esa\Pictures\_DATASET\unetpusbin\result_fold1"
files = os.listdir(path)

#NFilesMax = len(files)
#idx = int(input("NFilesMax  %d "%(NFilesMax)))
idx = 40

img = cv2.imread(os.path.join(path,files[idx]))
img2 = img

M = img.shape[0]
N = img.shape[1]

#print("%d %d"%(M,N))
#print("--------------------------")

start_point = (0, 0) 
end_point = (N, M)
color = (0, 255, 0)
thickness = 1
#img = cv2.line(img, start_point, end_point, color, thickness)

img2 = img.copy()


NoOfDivision = 50
rowDivSpace = M//NoOfDivision
print("NoOfDivision %d rowDivSpace %d "%(NoOfDivision,rowDivSpace))

iLeft =[]
jLeft =[]

iRight =[]
jRight =[]

thisRow = 0

bordRatioTh = 0.5
errThr = 0.3

# for i in range((M//5),(M-rowDivSpace),rowDivSpace):
# 	start_point = (0,i)
# 	end_point = (N,i)
# 	color2 = (0, 0, 255) # BGR format
# 	thisRow = 0
# 	for j in range(0,(N//3),1):
# 		bordStat = isBorder(img,i,j,5)
# 		#print("i : %d  j : %d   %3.4f "%(i,j,bordStat))
# 		err = abs (bordRatioTh - abs(bordStat)/255)
		
# 		if (err<=errThr) and (j>1) and (j<(N//3)) and (thisRow==0):
# 			#print("iLeft  %d %d"%(i,j))
# 			img2 = makeDot(img2, i, j)
# 			iLeft.append(i)
# 			jLeft.append(j)
# 			thisRow = 1
        

for i in range((M//5),(M-rowDivSpace),rowDivSpace):
	start_point = (0,i)
	end_point = (N,i)
	color2 = (0, 0, 255) # BGR format
	thisRow = 0

	for j in range(0,(N//3),1):
		j2 = (N-1-5) - j
		bordStat2 = isBorder(img,i,j2,5)
		err2 = abs(bordRatioTh -abs(bordStat2/255)) 
        
		if (err2<errThr) and (j2>(2*N//3)) and (j<N)and (thisRow==0):
			img2 = makeDot2(img2, i, j2)
			iRight.append(i)
			jRight.append(j2)
			thisRow = 1



#		print("%d %d  %4.3f"%(i,j,bordStat))




	img = cv2.line(img, start_point, end_point, color2, thickness)





#print("--------------------------")
N = len(iLeft)
#for i in range(N):
	#print("%d %d "%(iLeft[i],jLeft[i]))



#img2 = drawAPolyLine(iLeft, jLeft,img2,0)
img2 = drawAPolyLine(iRight, jRight,img2,1)






cv2.imshow("["+str(idx)+"] --- (2)"+files[idx], img2)


print("Done!")

cv2.waitKey(0)
cv2.destroyAllWindows()
