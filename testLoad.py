import keras
import os
from keras.models import model_from_json

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

