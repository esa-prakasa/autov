import numpy as np
import cv2
import os


os.system("cls")

path = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\vanish\\"
imageFile = "vanish.jpg"


img = cv2.imread(path+imageFile)
img = cv2.blur(img,(3,3))

M = img.shape[0]
N = img.shape[1]

N2 = N//2

samp = img[(M-10):(M-1),(N2-4):(N2+5),1]

Ms = (samp.shape[0])
Ns = (samp.shape[1])

accI = 0
for i in range(Ms):
	for j in range(Ns):
		accI = accI + samp[i,j]
thr = round(accI/(Ms*Ns))

print(thr)

thrLo = int(thr - 5)
thrHi = int(thr + 5)


#imgBin = np.zeros((M,N),dtype = int)
imgBin = np.zeros((M, N,1), np.uint8)


Mi = round(M*0.6)
for i in range(Mi,M,1):
	for j in range(N):
		val = img[i,j,1]
		if (val<thrLo or val>thrHi):
			imgBin[i,j] = 0
		else:
			imgBin[i,j] = 255

cv2.imshow("Binary",imgBin)







lines = cv2.HoughLines(imgBin,3, np.pi/5,100)


#edges = imgBin.copy()
#edges = cv2.Canny(imgBin,50,60,apertureSize = 3)
#cv2.imshow('frame3- Edges',edges)

#lines = cv2.HoughLines(edges,100, np.pi/180,300)


for line in lines:
	rho,theta = line[0]
	a = np.cos(theta)
	b = np.sin(theta)
	x0 = a * rho
	y0 = b * rho

	x1 = int(x0 + 1000*(-b))
	y1 = int(y0 + 1000*(a))

	x2 = int(x0 - 1000*(-b))
	y2 = int(y0 - 1000*(a))

	cv2.line(img, (x1, y1), (x2, y2), (0, 0, 255), 1)
	cv2.imshow('Image',img)

cv2.waitKey(0)
cv2.destroyAllWindows()

