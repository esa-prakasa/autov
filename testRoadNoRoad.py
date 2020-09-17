import cv2
import os
import matplotlib.pyplot as plt
import numpy as np

from keras.models import model_from_json


os.system("cls")

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


#x = np.array([[0.9, 0.9, 200, 200, 200]])

#x = np.array([[0.608,	0.267,	0.5,	0.48,	0.44]])

x = np.array([[0.608],	[0.326],	[0.5],	[0.47],	[0.44]])

x = np.array([[0.196],	[0.884],	[0.23],	[0.26],	[0.13]])

x = np.array([ [0.247],	[0.39],	[0.7],	[0.74],	[0.77] ])  ## class 0

x = np.array([ [0.866],	[0.785], [0.35], [0.34],	[0.31] ])  ## class 1
x = np.array([ [0.866],	[0.483], [0.3], [0.3], [0.29] ])  ## class 1


x = np.array([ [0.289],	[0.709], [0.27],	[0.27],	[0.12] ])  ## class 0



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