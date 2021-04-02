import os
import shutil
import cv2

os.system('cls')

rootPath =r'C:\Users\INKOM06\Pictures\roadDataset\pusbin1130segment'



imgPath = os.path.join(rootPath,'img')
maskPath = os.path.join(rootPath,'bw')
trainPath = os.path.join(rootPath,'train')

imgList = os.listdir(imgPath)
maskList = os.listdir(maskPath)

#for i in range(5):
#	print(imgList[i][:-4])

#for i in range(5):
#	print(maskList[i][:-4])


# create folder
'''
os.chdir(trainPath)
for folderName in imgList:
	folderName = folderName[:-4]
	print(folderName)
	if not os.path.exists(folderName):
		os.makedirs(folderName)
'''


# create sub folder images and mask
'''
folderList =os.listdir(trainPath)
print(folderList)

for folderName in folderList:
	folderPath = os.path.join(trainPath,folderName)
	print(folderPath)
	os.chdir(folderPath)
	if not os.path.exists('images'):
		os.makedirs('images')
	if not os.path.exists('mask'):
		os.makedirs('mask')
'''

folderList =os.listdir(trainPath)
N = len(folderList)
for i in range (N):
	#print(folderList[i])
	#fName = folderList[i]
	#img = cv2.imread(os.path.join(imgList[i]))
	
	#print(os.path.join(imgPath,imgList[i]))
	#print(os.path.join(maskPath,maskList[i]))

	img = cv2.imread(os.path.join(imgPath,imgList[i]))
	mask = cv2.imread(os.path.join(maskPath,maskList[i]))


folderList =os.listdir(trainPath)
for i in range(2) :#range(len(folderList)):
	img = cv2.imread(os.path.join(imgPath,imgList[i]))
	mask = cv2.imread(os.path.join(maskPath,maskList[i]))


	imgTargetPath = trainPath+"\\"+folderList[i]+"\\images\\"+imgList[i] 
	maskTargetPath = trainPath+"\\"+folderList[i]+"\\mask\\"+imgList[i] 

	cv2.imwrite(imgTargetPath,img)
	cv2.imwrite(maskTargetPath,mask)
	

	print(imgTargetPath+" has been saved")	
	print(maskTargetPath+" has been saved")