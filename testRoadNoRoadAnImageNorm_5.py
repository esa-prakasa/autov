import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


os.system("cls")


def classify(iNorm, jNorm, rNorm, gNorm, bNorm):

	x = np.array([ [iNorm],	[jNorm], [rNorm], [gNorm],	[bNorm] ])  ## class 0

	xT = np.array([ x[0], x[1], x[2], x[3], x[4] ])
	xT = np.transpose(xT)
	results = model.predict(xT)

	probVal = results[0][0]
#	probTh = 0.3
	probTh = 0.5
	if (probVal>probTh):
		#print("Class 1")
		classVal = 255 #1

	if (probVal<=probTh):
		#print("Class 0")
		classVal = 0


	return classVal




os.system("cls")

modelPath = r'C:\Users\INKOM06\Pictures\roadDataset\oregon_us\roads_annotated\ds\_json'

fileList = os.listdir(modelPath)

idx = 0
for fileNm in fileList:
	print("%d   %s"%(idx,fileNm))
	idx = idx + 1


idx = int(input("Which model that you want? "))

jsonPath = os.path.join(modelPath,fileList[idx])
print(jsonPath)



json_file = open(jsonPath, 'r') 
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.summary()



idx = 0
for fileNm in fileList:
	print("%d   %s"%(idx,fileNm))
	idx = idx + 1

idx = int(input("Which weights that you want to load? "))
hdfPath = os.path.join(modelPath,fileList[idx])
model.load_weights(hdfPath)











imgIdx = int(input("Image index please "))

#imgPath = r'C:\Users\INKOM06\Pictures\roadDataset\oregon_us\imgmask'

imgPath = r'C:\Users\INKOM06\Pictures\roadDataset\oregon_us\rgb'
imgPath = r'C:\Users\INKOM06\Pictures\roadDataset\oregon_us\rgbfull'
imgFiles = os.listdir(imgPath)
imgPath = os.path.join(imgPath,imgFiles[imgIdx])





img = cv2.imread(imgPath)


ratio = 0.3

M = img.shape[0]
N = img.shape[1]


img = cv2.resize(img, (int(N*ratio),int(M*ratio)),interpolation = cv2.INTER_AREA)

M = img.shape[0]
N = img.shape[1]


imgBin = np.zeros((M,N), dtype = np.uint8)


import pandas as pd
data = {'i':  [], 
'j': [], 
'r': [], 
'g': [], 
'b': [], 
}


# for i in range(M):
# 	for j in range(N):
# 		#print("%d  %d  %d  %d  %d"%(i,j,img[i,j,2],img[i,j,1],img[i,j,0]))
# 		iNorm = i/M - 0.5
# 		jNorm = j/N - 0.5
# 		rNorm = float(img[i,j,2]/255) - 0.5
# 		gNorm = float(img[i,j,1]/255) - 0.5
# 		bNorm = float(img[i,j,0]/255) - 0.5

# 		#print(rNorm)

# 		data['i'].append(iNorm)
# 		data['j'].append(iNorm)
# 		data['r'].append(rNorm)
# 		data['g'].append(gNorm)
# 		data['b'].append(bNorm)


		
# X = pd.DataFrame (data, columns = ['i','j','r','g','b'])

#from sklearn.preprocessing import StandardScaler
#sc = StandardScaler()
#X = sc.fit_transform(X)

#print(X.iloc[0,0:4])




#print(X[0][0])




NumDt = M*N #len(X)

idxDt = 0
for i in range(M):
	for j in range(N):
		#iNorm = X.iloc[idxDt][0]
		#jNorm = X.iloc[idxDt][1]
		#rNorm = X.iloc[idxDt][2]
		#gNorm = X.iloc[idxDt][3]
		#bNorm = X.iloc[idxDt][4]
		

		iNorm = i/M - 0.5
		jNorm = j/N - 0.5
		rNorm = float(img[i,j,2]/255) - 0.5
		gNorm = float(img[i,j,1]/255) - 0.5
		bNorm = float(img[i,j,0]/255) - 0.5



		classVal = classify(iNorm, jNorm, rNorm, gNorm, bNorm)
		imgBin[i,j] = classVal
		idxDt = idxDt + 1

		print("%3.2f"%(idxDt*100/NumDt))

		#print("%f  %f  %f  %f  %f ---> %d"%(iNorm, jNorm, rNorm, gNorm, bNorm, classVal))





img2 = img

M = img2.shape[0] 
N = img2.shape[1] 


for i in range(M):
	for j in range(N):
		if imgBin[i,j]==255:
			img2[i,j,1] = 0



#cv2.imshow("img", img)
#cv2.imshow("img Bin", imgBin)


img2 = cv2.resize(img2, (int(N/ratio),int(M/ratio)),interpolation = cv2.INTER_AREA)
cv2.imshow("img 2", img2)


cv2.waitKey(0)
cv2.destroyAllWindows()
