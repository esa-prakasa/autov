# augmented the images to vary the dataset.
# This augmentation aims to increase the datset size by applying several variations.

import os
import cv2
# example of horizontal shift image augmentation
from numpy import expand_dims
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
os.system("cls")

srcFolder = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\temp\\B\\"
tgtFolder = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\temp\\B_aug\\"

srcFiles = os.listdir(srcFolder)
tgtFiles = []
for i in range(len(srcFiles)):
	#print(srcFiles[i])
	fNm = srcFiles[i]
	fNm = fNm[:-4]
	tgtFiles.append(fNm)


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

	for i in range(42):
		batch = it.next()
		image = batch[0].astype('uint8')
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

		idxS = str(100+i)
		idxS = idxS[1:]

		fileName = tgtFiles[idx]+"_"+idxS+".jpg"
		pathToSave = tgtFolder+fileName
		print(pathToSave)
		cv2.imwrite(pathToSave, image) 

