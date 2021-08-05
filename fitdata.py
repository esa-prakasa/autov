import cv2
import os
import numpy as np

#img0 = cv2.imread(r'C:\Users\INKOM06\Pictures\roadDataset\oregon_us\roads_annotated\ds\bw\00570.png')
img0 = cv2.imread(r'C:\Users\INKOM06\Pictures\_autonomous\fit\right.png')
cv2.imshow("Img0", img0)


M = img0.shape[0]
N = img0.shape[1]


#for i in range(M):









os.system("cls")


img = cv2.imread(r'C:\Users\INKOM06\Pictures\_autonomous\fit\line.png')

#cv2.imshow("Img", img)


M = img.shape[0]
N = img.shape[1]

imgc = np.zeros([M,N,3])

imgc[:,:,0] = img[:,:,0]
imgc[:,:,1] = img[:,:,0]
imgc[:,:,2] = img[:,:,0]


x = np.array([])
y = np.array([])

for i in range(M):
	for j in range(N):
		if (img[i,j,0]>=200):
			#print("i: %d  j: %d"%(i,j))
			x = np.append(x, j)
			y = np.append(y, i)

order = 3
z = np.polyfit(x,y,order)


def computeY(j,z,order):
	res = 0
	for i in range(order+1):
		res = res + z[i]*(j**(order-i))
	#print(res)
	return res







N = len(x)
for idx in range(N):
	j = float(x[idx])
	#yR = z[0]*j*j + z[1]*j + z[2]
	yR = computeY(j,z,order)
	i = int(yR)

	j2 = int(j)
	imgc[i,j2,0] = 255
	imgc[i,j2,1] = 0
	imgc[i,j2,2] = 255


cv2.imshow("imgc", imgc)


print(z)




'''

x = np.array([0.0, 1.0, 2.0, 3.0,  4.0,  5.0])
#y = np.array([0.0, 0.8, 0.9, 0.1, -0.8, -1.0])
y = np.array([0.0, 2.0, 4.0, 6.0, 8.3, 10.0])
z = np.polyfit(x, y, 1)

print(z)
'''

cv2.waitKey(0)
cv2.destroyAllWindows()
