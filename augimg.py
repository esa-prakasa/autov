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
		rotation_range=30,
		brightness_range=[0.5,1.0],
		zoom_range=[0.5,1.0])
	it = datagen.flow(samples, batch_size=1)

	for i in range(30):
		batch = it.next()
		image = batch[0].astype('uint8')
		image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

		idxS = str(100+i)
		idxS = idxS[1:]

		fileName = tgtFiles[idx]+"_"+idxS+".jpg"
		pathToSave = tgtFolder+fileName
		print(pathToSave)
		cv2.imwrite(pathToSave, image)


# show the figure
#pyplot.show()







#fileName = "0__260_540_0.59.jpg"

'''

# load the image
img = load_img("C:\\Users\\INKOM06\\Pictures\\fontdata\\6\\"+fileName)
# convert to numpy array
data = img_to_array(img)
# expand dimension to one sample
samples = expand_dims(data, 0)
# create image data augmentation generator
#datagen = ImageDataGenerator(width_shift_range=[-200,200])
shftPix = 3
datagen = ImageDataGenerator(
	height_shift_range=[-shftPix,shftPix],
	width_shift_range=[-shftPix,shftPix],
	rotation_range=30,
	brightness_range=[0.7,1.0],
	zoom_range=[0.5,1.0])

# prepare iterator
it = datagen.flow(samples, batch_size=1)
# generate samples and plot
for i in range(20):
	# define subplot
	#pyplot.subplot(330 + 1 + i)
	# generate batch of images
	batch = it.next()
	# convert to unsigned integers for viewing
	image = batch[0].astype('uint8')
	# plot raw pixel data
	pyplot.imshow(image)

	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

	fileName = str(i)+".jpg"
	pathToSave = "C:\\Users\\INKOM06\\Pictures\\fontdata\\6\\aug\\"+fileName
	print(pathToSave)
	cv2.imwrite(pathToSave, image)


# show the figure
#pyplot.show()

'''