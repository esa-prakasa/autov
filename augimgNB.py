import os
import cv2
# example of horizontal shift image augmentation
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import numpy as np


os.system("cls")

srcFolder = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\temp\\NB\\"
tgtFolder = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\temp\\NB_aug\\"

srcFiles0 = os.listdir(srcFolder)
NscMax = len(srcFiles0)

permIdx = np.random.permutation(NscMax)


print(permIdx[0:10])


NscImg = 90
srcFiles = []
tgtFiles = []
for i in range(NscImg): 
	srcFiles.append(srcFiles0[permIdx[i]])
	fNm = srcFiles[i]
	fNm = fNm[:-4]
	tgtFiles.append(fNm)
	print(str(permIdx[i])+"  "+srcFiles[i]+"  "+tgtFiles[i])





for idx in range(len(srcFiles)):
	print(tgtFiles[idx])
	img = load_img(srcFolder+srcFiles[idx])
	data = img_to_array(img)
	samples = expand_dims(data, 0)
	shftPix = 3

	datagen = ImageDataGenerator(
		height_shift_range=[-shftPix,shftPix],
		width_shift_range=[-shftPix,shftPix],
		rotation_range=15,
		brightness_range=[0.5,1.0],
		zoom_range=[0.5,1.0])
	it = datagen.flow(samples, batch_size=1)

	for i in range(11):
		batch = it.next()
		image = batch[0].astype('uint8')
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

		idxS = str(1000+i)
		idxS = idxS[1:]

		fileName = tgtFiles[idx]+"_"+idxS+".jpg"
		pathToSave = tgtFolder+fileName
		print(pathToSave)
		cv2.imwrite(pathToSave, image)

