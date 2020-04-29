import cv2
import os
import matplotlib.pyplot as plt
import time
import numpy as np

start_time = time.time()

os.system("cls")


def getFileArray(fileName):
	textFile = "C:\\Users\\INKOM06\\Pictures\\roadlane-detection-evaluation-2013\\training\\textfolder\\"+fileName
	imgFilesPath = []

	with open(textFile) as fp:
   		line = fp.readline()
   		cnt = 1
   		while line:
   			line = fp.readline()
   			pathStr = line.strip()
   			imgFilesPath.append(pathStr)
   			#print(str(cnt)+": "+imgFilesPath[cnt-1])
   			cnt += 1
	cnt = cnt-1

	return imgFilesPath, cnt

def getDataPoints(boxMat):
	M = boxMat.shape[0]
	N = boxMat.shape[1]

	bl = []
	gr = []
	rd = []

	for i in range(M):
		if (i%2)==0:
			for j in range (N):
				bl.append(boxMat[i,j,0])
				gr.append(boxMat[i,j,1])
				rd.append(boxMat[i,j,2])
		if (i%2)==1:
			for j in range ((N-1),-1,-1):
				bl.append(boxMat[i,j,0])
				gr.append(boxMat[i,j,1])
				rd.append(boxMat[i,j,2])
	return rd,gr,bl

def reshapeRGB(rd,gr,bl):
	N = len(rd)
	N2 = 2*N
	#rgb = np.zeros((N*3, 1, 1), dtype = "uint8")
	rgb = [0]*(N*3)
	for i in range(N):
		rgb[i]    = rd[i]
		rgb[i+N]  = gr[i]
		rgb[i+N2] = bl[i]

	rgbStr =""
	for i in range(N*3):
		rgbStr = rgbStr+str(rgb[i])+","
	#rgbStr = rgbStr[:-1]

	return rgb,rgbStr





[oriFilesPath,cnt] = getFileArray("ori.txt")
print(cnt)
print("============================================")
[labFilesPath,cnt] = getFileArray("lab.txt")
print(cnt)

ratio = 0.2
filterSize = 5
fd = filterSize//2
noPixArea = filterSize*filterSize

for idx in range(1):

	oriImg = cv2.imread(oriFilesPath[idx])
	labImg = cv2.imread(labFilesPath[idx])

	M = labImg.shape[0]
	N = labImg.shape[1]
	#print("M: %d and N: %d "%(M,N))

	oriImg = cv2.resize(oriImg, (int(ratio*N),int(ratio*M)))
	labImg = cv2.resize(labImg, (int(ratio*N),int(ratio*M)))

	M = labImg.shape[0]
	N = labImg.shape[1]
	#print("M: %d and N: %d "%(M,N))

	oriImg2 = oriImg[(M//2):M,:,:]
	labImg2 = labImg[(M//2):M,:,:]

	M2 = labImg2.shape[0]
	N2 = labImg2.shape[1]

	#cv2.imshow(("ori "+str(idx)),oriImg2)
	#cv2.imshow(("lab "+str(idx)),labImg2)

	bwImg = np.zeros((M2, N2, 3), dtype = "uint8")

	totalPix = 0

	for i in range(fd,(M2-fd),1):
		for j in range(fd,(N2-fd),1):
			roadPix = 0

			locStr = str(i)+","+str(j)+"--"

			boxMat =np.zeros((filterSize, filterSize, 3), dtype = "uint8")
			for k in range(-fd,(fd+1),1):
				for l in range(-fd,(fd+1),1):
					bRef = labImg2[(i+k),(j+l),0]
					boxMat[(k+fd),(l+fd),0] = oriImg2[(i+k),(j+l),0]
					boxMat[(k+fd),(l+fd),1] = oriImg2[(i+k),(j+l),1]
					boxMat[(k+fd),(l+fd),2] = oriImg2[(i+k),(j+l),2]

					if (bRef==255):   ##>>> ROAD
						roadPix = roadPix + 1

			roadVal = roadPix/noPixArea

			if (roadVal>=0.5):
				for k in range(3):
					bwImg[i,j,k] = 255

					[rd,gr,bl] = getDataPoints(boxMat)
					[nRGB,rgbStr] = reshapeRGB(rd,gr,bl)
					laneClass = "1"
					rgbStr=locStr + rgbStr + laneClass 
					print(rgbStr)
					totalPix +=1 

			if (roadVal<0.5):
				for k in range(3):
					bwImg[i,j,k] = 0

					[rd,gr,bl] = getDataPoints(boxMat)
					[nRGB,rgbStr] = reshapeRGB(rd,gr,bl)
					laneClass = "0"
					rgbStr=locStr + rgbStr + laneClass 

					print(rgbStr)
					totalPix +=1 

	#cv2.imshow(("BW "+str(idx)),bwImg)



print("Total pixel is %d"%totalPix)




'''
#path = "C:\\Users\\INKOM06\\Pictures\\jagung2020\\largeDataSet\\true\\"
labelPath ="C:\\Users\\INKOM06\\Pictures\\roadlane-detection-evaluation-2013\\data_road\\training\\gt_image_2\\"
oriPath ="C:\\Users\\INKOM06\\Pictures\\roadlane-detection-evaluation-2013\\data_road\\training\\image_2\\"
pathToSave = "C:\\Users\\INKOM06\\Documents\\[0--KEGIATAN-Ku-2020\\2020.01-006-Autonomous Vehicle Project\\pixSegmentation202004\\roadreg\\"

#fileTxtName = "rat30pct0414.txt"
#nonRoadName = "nonroad30pct_0414.txt" 
NofImage = 200
ratio = 0.2
ratioPct = int(ratio*100)

csvRoadFileNm = "road"+str(ratioPct)+"pct"+"_"+str(NofImage)+".csv"


#logFile = open((pathToSave+fileTxtName),"w+")
#logFileNR = open((pathToSave+nonRoadName),"w+")
csvRoadFile = open((pathToSave+"csvfiles\\"+csvRoadFileNm),"w+")



outVal = ("No, i, j,  r,  g, b, class")
csvRoadFile.write(outVal+"\n")




for idx in range(30):
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

	for i in range(M2):
		for j in range(N2):
			bRef = labImg2[i,j,0]
			b = oriImg2[i,j,0]
			g = oriImg2[i,j,1]
			r = oriImg2[i,j,2]
			
			if (bRef==0):  ##>>> Non Road
				roadClass = 0
			if (bRef==255):   ##>>> ROAD
				roadClass = 1

			outVal = ("%d, %d, %d,  %d,  %d, %d, %d"%(idx,i, j, r, g, b, roadClass))
			print(outVal)

			csvRoadFile.write(outVal+"\n")

csvRoadFile.close()
'''

deltaTime = time.time() - start_time 
print("--- %5.5s seconds ---" % (deltaTime))


cv2.waitKey()
cv2.destroyAllWindows()
