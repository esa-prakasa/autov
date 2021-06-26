import cv2
import os
import numpy as np

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


os.system("cls")
path = r"C:\Users\Esa\Pictures\_DATASET\unetpusbin\result_fold1"
files = os.listdir(path)

idx = 400
img = cv2.imread(os.path.join(path,files[idx]))
img2 = img

M = img.shape[0]
N = img.shape[1]

print("%d %d"%(M,N))
print("--------------------------")

start_point = (0, 0) 
end_point = (N, M)
color = (0, 255, 0)
thickness = 1
#img = cv2.line(img, start_point, end_point, color, thickness)

img2 = img.copy()


NoOfDivision = 20
rowDivSpace = M//NoOfDivision
print("NoOfDivision %d rowDivSpace %d "%(NoOfDivision,rowDivSpace))

iLeft =[]
jLeft =[]

for i in range(0,(M-rowDivSpace),rowDivSpace):
	start_point = (0,i)
	end_point = (N,i)
	color2 = (0, 0, 255) # BGR format

	for j in range(0,(N//2),1):
		bordStat = isBorder(img,i,j,5)

		err = abs (0.5 - abs(bordStat)/255)

#		if (bordStat>=102) and (bordStat<=153):
		if (err<=0.05):
			#print(bordStat)
			print("%d %d"%(i,j))
			img2 = makeDot(img2, i, j)
			iLeft.append(i)
			jLeft.append(j)
		
#		print("%d %d  %4.3f"%(i,j,bordStat))


	img = cv2.line(img, start_point, end_point, color2, thickness)





print("--------------------------")
N = len(iLeft)
for i in range(N):
	print("%d %d "%(iLeft[i],jLeft[i]))


cv2.imshow("["+str(idx)+"] --- "+files[idx], img)
cv2.imshow("["+str(idx)+"] --- (2)"+files[idx], img2)

cv2.waitKey(0)
cv2.destroyAllWindows()
