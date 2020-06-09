
import os
import cv2
import numpy as np

os.system("cls")

path ="C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\Purbaleunyi_Highway_1\\ds\\_B3\\"

files = os.listdir(path)

N = (len(files))
Nf = N

Size = []

for i in range(N):
	img = cv2.imread(path+files[i])

	M = img.shape[0]
	N = img.shape[1]

	print("%d  [%d , %d]"%(i,M,N))

	dim = str(M)+" x "+str(N)
	if dim not in Size:
		Size.append(dim)
		print(Size)


print(Size)


#Size = ['30 x 30', '30 x 29', '30 x 28', '30 x 27', '30 x 26', '30 x 25', '30 x 24', '30 x 22', '30 x 21', '30 x 23', '30 x 20', '29 x 30', '28 x 30', '27 x 30', '26 x 30', '25 x 30', '24 x 30', '23 x 30', '22 x 30', '21 x 30', '20 x 30', '19 x 30', '18 x 30', '17 x 30']

Ns = (len(Size))

acc = np.zeros(Ns,dtype=np.int64)

for i in  range(Nf):
	img = cv2.imread(path+files[i])

	M = img.shape[0]
	N = img.shape[1]

	print("%d  [%d , %d]"%(i,M,N))

	dim = str(M)+" x "+str(N)

	for j in range(Ns):
		if dim == Size[j]:
			acc[j] = acc[j] + 1

os.system("cls")
print("    \n")
for i in range(Ns):
	print("    %s : %d  --> %2.1f%%"%(Size[i],acc[i],(acc[i]*100/Nf)))

















	