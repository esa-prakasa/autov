import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

from keras.models import model_from_json
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()


os.system("cls")


def classify(iNorm, jNorm, rNorm, gNorm, bNorm):
	#totVal = (iNorm + jNorm + rNorm + gNorm + bNorm)/5
	#if (totVal>0.5):
	#	classVal = 255
	#else:
	#	classVal = 0

	x = np.array([ [iNorm],	[jNorm], [rNorm], [gNorm],	[bNorm] ])  ## class 0

	xs = sc.fit_transform(x)

	#print(xs)

	xT = np.array([ xs[0], xs[1], xs[2], xs[3], xs[4] ])
	xT = np.transpose(xT)
	#print(xT)

#result = model.predict(xs)
	results = model.predict(xT)

#print(xs)
#print(result)
#print(results)

	probVal = results[0][0]
	if (probVal>=0.5):
		#print("Class 1")
		classVal = 255 #1

	if (probVal<0.5):
		#print("Class 0")
		classVal = 0


	return classVal



# Load model to the workspace
modelPath = r'C:\Users\INKOM06\Pictures\roadDataset\oregon_us\roads_annotated\ds\_json\saved'

modelFiles = os.listdir(modelPath)
idx = 0
for fileName in modelFiles:
	jsonType = fileName[-2:]
	if (jsonType == "on"):
		print("%d   %s"%(idx,fileName))
	idx = idx + 1


jsonIdx = int(input("Which json index that you want? "))
jsonPath = os.path.join(modelPath,modelFiles[jsonIdx])
print(jsonPath)

json_file = open(jsonPath, 'r') 
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

model.summary()

h5Path = jsonPath[:-4]+"h5"
print(h5Path)





x = np.array([[0.608],	[0.326],	[0.5],	[0.47],	[0.44]])
x = np.array([[0.196],	[0.884],	[0.23],	[0.26],	[0.13]])
x = np.array([ [0.247],	[0.39],	[0.7],	[0.74],	[0.77] ])  ## class 0
x = np.array([ [0.866],	[0.785], [0.35], [0.34],	[0.31] ])  ## class 1
x = np.array([ [0.866],	[0.483], [0.3], [0.3], [0.29] ])  ## class 1
x = np.array([ [0.289],	[0.709], [0.27],	[0.27],	[0.12] ])  ## class 0




#imgPath = r'C:\Users\INKOM06\Pictures\roadDataset\oregon_us\rgb'
#imgFiles = os.listdir(imgPath)
#imgIdx  = int(input("Which image index that you want? Max: "))
#imgPath = os.path.join(imgPath,imgFiles[imgIdx])
#print(imgPath)


#imgPath = r'C:\Users\INKOM06\Pictures\roadDataset\oregon_us\rgb'

#imgPath = r'C:\Users\INKOM06\Pictures\roadDataset\oregon_us\imgmask'
#imgFiles = os.listdir(imgPath)
#imgIdx  = int(input("Which image index that you want? Max: "))
#imgIdx = 0
#imgPath = os.path.join(imgPath,imgFiles[imgIdx])
#print(imgPath)

#imgPath = r'C:\Users\INKOM06\Pictures\roadDataset\oregon_us\imgmask'
#imgPath = os.path.join(imgPath,"00070.png")
#imgPath = os.path.join(imgPath,"00012.png")


# imgPath = r'C:\Users\INKOM06\Pictures\roadDataset\oregon_us\rgb'
# imgPath = os.path.join(imgPath,"00070.png")

imgIdx = int(input("Image index please "))

imgPath = r'C:\Users\INKOM06\Pictures\roadDataset\oregon_us\imgmask'
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


for i in range(M):
	for j in range(N):
		#print("%d  %d  %d  %d  %d"%(i,j,img[i,j,2],img[i,j,1],img[i,j,0]))
		iNorm = i/M
		jNorm = j/N
		rNorm = img[i,j,2]/255
		gNorm = img[i,j,1]/255
		bNorm = img[i,j,0]/255

		data['i'].append(iNorm)
		data['j'].append(iNorm)
		data['r'].append(rNorm)
		data['g'].append(gNorm)
		data['b'].append(bNorm)


		
X = pd.DataFrame (data, columns = ['i','j','r','g','b'])

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)

print(X)

print(X[0][0])

NumDt = len(X)

idxDt = 0
for i in range(M):
	for j in range(N):
		iNorm = X[idxDt][0]
		jNorm = X[idxDt][1]
		rNorm = X[idxDt][2]
		gNorm = X[idxDt][3]
		bNorm = X[idxDt][4]
		classVal = classify(iNorm, jNorm, rNorm, gNorm, bNorm)
		imgBin[i,j] = classVal
		idxDt = idxDt + 1

		print("%3.2f"%(idxDt/NumDt))

		#print("%f  %f  %f  %f  %f ---> %d"%(iNorm, jNorm, rNorm, gNorm, bNorm, classVal))




cv2.imshow("img", img)
cv2.imshow("img Bin", imgBin)









'''

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
xs = sc.fit_transform(x)

print(xs)


xT = np.array([ xs[0], xs[1], xs[2], xs[3], xs[4] ])
xT = np.transpose(xT)
print(xT)

#result = model.predict(xs)
results = model.predict(xT)

#print(xs)
#print(result)
print(results)

probVal = results[0][0]
if (probVal>=0.5):
	print("Class 1")

if (probVal<0.5):
	print("Class 0")

'''

cv2.waitKey(0)
cv2.destroyAllWindows()
