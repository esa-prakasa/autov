#input image
#output histogram

import numpy as np
import cv2
import os

#path = "C:\\Users\\INKOM06\\Pictures\\fontdata\\6\\"
path = "C:\\Users\\INKOM06\\Pictures\\rgb\\"

#fileNm = "20190509_190049.jpg"

fileNm = "test.jpg"





def lbpFilter(img, fSz):
	fSz2 = fSz//2

	M = img.shape[0]
	N = img.shape[1]


	img2 = np.zeros((fSz,fSz), np.uint8)


	for i in range((fSz2),(M-1-fSz2),1):
		for j in range((fSz2),(N-1-fSz2),1):

			img2 = img[(i-fSz2):(i+fSz2),(j-fSz2):(j+fSz2)]
			mm = img2.shape[0]
			nn = img2.shape[1]
			ref = img2[fSz2,fSz2]
			#print(ref)

			H = 0
			#print("%d    %d"%(mm,nn))

			if img2[0,0]>=0:
				H = H + 2**0
'''

			if img2[0,1]>=0:
				H = H + 2**1

			if img2[0,2]>=0:
				H = H + 2**2

			if img2[1,2]>=0:
				H = H + 2**3

			if img2[2,2]>=0:
				H = H + 2**4

			if img2[2,1]>=0:
				H = H + 2**5

			if img2[2,0]>=0:
				H = H + 2**6

			if img2[1,0]>=0:
				H = H + 2**7
'''

			## 0  1  2
			## 7  R  3
			## 6  5  4


#			for k in range(mm):
#				for l in range(nn):
	#return ref











#	return outImg
ratio = 0.5

img = cv2.imread(path+fileNm)

M = img.shape[0]
N = img.shape[1]
img = cv2.resize(img, (int(ratio*N),int(ratio*M)))

R = img[:,:,2]
G = img[:,:,1]
B = img[:,:,0]



rgb = np.hstack((R,G,B))
    

#cv2.imshow("Original RGB",img)
#cv2.imshow("RGB",rgb)

lbpFilter(R,3)



cv2.waitKey(0)
cv2.destroyAllWindows()
