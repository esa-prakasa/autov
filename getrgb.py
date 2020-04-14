import cv2
import os
import matplotlib.pyplot as plt
import time

start_time = time.time()

os.system("cls")



#path = "C:\\Users\\INKOM06\\Pictures\\jagung2020\\largeDataSet\\true\\"
labelPath ="C:\\Users\\INKOM06\\Pictures\\roadlane-detection-evaluation-2013\\data_road\\training\\gt_image_2\\"
oriPath ="C:\\Users\\INKOM06\\Pictures\\roadlane-detection-evaluation-2013\\data_road\\training\\image_2\\"
pathToSave = "C:\\Users\\INKOM06\\Documents\\[0--KEGIATAN-Ku-2020\\2020.01-006-Autonomous Vehicle Project\\pixSegmentation202004\\roadreg\\"

fileTxtName = "rat30pct0414.txt"
nonRoadName = "nonroad30pct_0414.txt" 

logFile = open((pathToSave+fileTxtName),"w+")
logFileNR = open((pathToSave+nonRoadName),"w+")





for idx in range(289):
#idx = int(input("Image index that needs to be analysed? "))
#idx = 31



	labFiles = []
# r=root, d=directories, f = files
	for r, d, f in os.walk(labelPath):
		for file in f:
			labFiles.append(os.path.join(r, file))


	oriFiles = []
# r=root, d=directories, f = files
	for r, d, f in os.walk(oriPath):
		for file in f:
			oriFiles.append(os.path.join(r, file))



	labImg = cv2.imread(labFiles[idx])
	oriImg = cv2.imread(oriFiles[idx])

	M = labImg.shape[0]
	N = labImg.shape[1]
	#print("M: %d and N: %d "%(M,N))
	ratio = 0.3

	labImg = cv2.resize(labImg, (int(ratio*N),int(ratio*M)))
	oriImg = cv2.resize(oriImg, (int(ratio*N),int(ratio*M)))

	M = labImg.shape[0]
	N = labImg.shape[1]
	#print("M: %d and N: %d "%(M,N))

	oriImg2 = oriImg[(M//2):M,:,:]
	labImg2 = labImg[(M//2):M,:,:]

	M2 = labImg2.shape[0]
	N2 = labImg2.shape[1]


	roadReg = oriImg2.copy()

#cv2.imshow("labImg 2", labImg2)
#cv2.imshow("oriImg 2", oriImg2)

	bRAc = 0
	gRAc = 0
	rRAc = 0
	nPix = 0

	NbRAc = 0
	NgRAc = 0
	NrRAc = 0
	NnPix = 0


	for i in range(M2):
		for j in range(N2):
			b = labImg2[i,j,0]
			g = labImg2[i,j,1]
			r = labImg2[i,j,2]
			if (b==0):
				roadReg[i,j,0] = 0
				roadReg[i,j,1] = 0
				roadReg[i,j,2] = 0
			if (b!=0):
				bR = roadReg[i,j,0]
				gR = roadReg[i,j,1]
				rR = roadReg[i,j,2]
				nPix += 1
				bRAc += bR
				gRAc += gR
				rRAc += rR
			if (b==0):
				NbR = oriImg2[i,j,0]
				NgR = oriImg2[i,j,1]
				NrR = oriImg2[i,j,2]
				NnPix += 1
				NbRAc += NbR
				NgRAc += NgR
				NrRAc += NrR

#			if ((nPix%100)==0):
#				print("%d %d %d %d %d %d"%(nPix,(i+M2),j,rR,gR,bR))


	bMean = bRAc/nPix
	gMean = gRAc/nPix
	rMean = rRAc/nPix

	NbMean = NbRAc/NnPix
	NgMean = NgRAc/NnPix
	NrMean = NrRAc/NnPix


	#print("Red Mean: %4.2f"%rMean)
	#print("Green Mean: %4.2f"%gMean)
	#print("Blue Mean: %4.2f"%bMean)


	#cv2.imshow("roadReg", roadReg)


	outlines = ("%d %d %4.1f %4.1f %4.1f"%(idx, nPix, rMean, gMean, bMean))
	outlinesN = ("%d %d %4.1f %4.1f %4.1f"%(idx, NnPix, NrMean, NgMean, NbMean))
	print(outlines)
	fullPathToSave = pathToSave+"road"+str(idx)+'.jpg'
	cv2.imwrite(fullPathToSave, roadReg)
	logFile.write(outlines+"\n")
	logFileNR.write(outlinesN+"\n")


#print(type(outlines))



logFile.close()
logFileNR.close()


'''


layImg =oriImg.copy()

M = layImg.shape[0]
N = layImg.shape[1]

print(M)
print(N)


for i in range(M):
	for j in range (N):
		b = labImg[i,j,0]
		g = labImg[i,j,1]
		r = labImg[i,j,2]
		if ((r==255)&(b==255)):
			layImg[i,j,0] = 0
			#layImg[i,j,1] = 0
			#layImg[i,j,2] = 0


b,g,r = cv2.split(oriImg)       # get b,g,r
rgb_oriImg = cv2.merge([r,g,b])     # switch it to rgb

b,g,r = cv2.split(layImg)       # get b,g,r
rgb_layImg = cv2.merge([r,g,b])     # switch it to rgb


#cv2.imshow("labImg ",labImg)
#cv2.imshow("oriImg ",oriImg)
#cv2.imshow("layImg ",layImg)


fig, axs = plt.subplots(3,1)
fig.suptitle("Lane area segmentation")
axs[0].imshow(rgb_oriImg)
axs[1].imshow(labImg)
axs[2].imshow(rgb_layImg)
plt.show()

'''
deltaTime = time.time() - start_time 
print("--- %5.5s seconds ---" % (deltaTime))


cv2.waitKey()
cv2.destroyAllWindows()
