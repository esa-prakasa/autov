import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras.callbacks import ModelCheckpoint

from keras.models import load_model
from keras.preprocessing import image


kfold = "_fold3"


rootPath  = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\dataset\\"+kfold+"\\train\\"
modelPath = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\dataset\\"+kfold+"\\xmodel\\"
modelName = "best_model_"+kfold+".h5"
model = load_model(modelPath+modelName)

testFolder =[]
testFolder.append("C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\dataset\\"+kfold+"\\test\\B\\")
testFolder.append("C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\dataset\\"+kfold+"\\test\\NB\\")

dimSz = 20
os.system("cls")
accuracy = []
for clsSelected in range (2):
	testedFolder = testFolder[clsSelected]

	files = os.listdir(testedFolder)
	NF = len(files)

	target_names = [item for item in os.listdir(rootPath) if os.path.isdir(os.path.join(rootPath, item))]

	correctCount = 0

	for i in range(NF): #range(Nb):
		test_image = image.load_img((testedFolder+files[i]), target_size =(dimSz, dimSz))
		test_image = image.img_to_array(test_image)
		test_image = np.expand_dims(test_image, axis = 0)
		result = int(model.predict(test_image))
		#print("%2.2f"%(result[0]))
		#classIdx = np.argmax(result)
		print(str(i)+" --- "+files[i]+"--"+str(clsSelected)+" >>> "+str(result))
		if (clsSelected==result):
			correctCount = correctCount + 1

	accuracy.append(correctCount/NF)

print("  ")
for clsSelected in range(2):	
	print("Class of "+target_names[clsSelected]+(" %3.2f"%(accuracy[clsSelected])))


cm = []
cm.append([accuracy[0], (1-accuracy[0])])
cm.append([(1-accuracy[1]), accuracy[1]])

print(cm)




