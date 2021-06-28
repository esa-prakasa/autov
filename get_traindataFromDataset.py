import os
import cv2
import time
import math

start_time = time.time()

os.system("cls")

bwPath = r'C:\Users\Esa\Pictures\_DATASET\pusbin\bw'
colPath = r'C:\Users\Esa\Pictures\_DATASET\pusbin\img'
savePath = r'C:\Users\Esa\Pictures\_DATASET\pusbin\_csv'

bwList  = os.listdir(bwPath)
colList = os.listdir(colPath)

ratio = float(input("Give resize ratio of the image: "))
deltaSamp = int(input("What is sampling distance in pixels, opt: 1, 5, 10, 20: "))


bwFilePath  = os.path.join(bwPath,bwList[0])
imgBW = cv2.imread(bwFilePath)
imgBW = cv2.cvtColor(imgBW, cv2.COLOR_BGR2GRAY)
M = imgBW.shape[0]
N = imgBW.shape[1]
m = math.floor(M*ratio)
n = math.floor(N*ratio)



NofFiles = len(bwList)
idxOfTrainFile = 0

#NofFiles = 3

print(NofFiles)



csvFileNm   = "csc_"+str(int(ratio*100))+"pct_samp_"+str(deltaSamp)+".csv"
csvFilePath = os.path.join(savePath,csvFileNm)
csvRoadFile = open(csvFilePath,"w+")
outVal = ("No, i, j,  r,  g, b, class")
csvRoadFile.write(outVal+"\n")

dataIdx = 0


for idxOfTrainFile in range (NofFiles):

	bwFilePath  = os.path.join(bwPath,bwList[idxOfTrainFile])
	colFilePath = os.path.join(colPath,colList[idxOfTrainFile])

	imgBW = cv2.imread(bwFilePath)
	imgBW = cv2.cvtColor(imgBW, cv2.COLOR_BGR2GRAY)
	ret,imgBW = cv2.threshold(imgBW,127,255,cv2.THRESH_BINARY)

	imgCol = cv2.imread(colFilePath)


	# Resize both images
	imgBW  = cv2.resize(imgBW, (n,m), interpolation = cv2.INTER_AREA) 
	imgCol = cv2.resize(imgCol, (n,m), interpolation = cv2.INTER_AREA) 


	M = imgBW.shape[0]
	N = imgBW.shape[1]

	for i in range(0,M,deltaSamp):
		for j in range(0,N,deltaSamp):
			 if imgBW[i,j] == 255:
			 	cl = 1
			 if imgBW[i,j] == 0:
			 	cl = 0

			 iR = i/M - 0.5
			 jR = j/N - 0.5
			 R = imgCol[i,j,2]/255 - 0.5
			 G = imgCol[i,j,1]/255 - 0.5
			 B = imgCol[i,j,0]/255 - 0.5
			 outVal = "%d, %1.3f, %1.3f, %1.2f, %1.2f, %1.2f, %d"%(dataIdx,iR,jR,R,G,B,cl) 
			 print("%d %d %d  ---- %s"%(idxOfTrainFile,i,j,outVal))

			 csvRoadFile.write(outVal+"\n")

			 dataIdx = dataIdx + 1



csvRoadFile.close()

deltaTime = time.time() - start_time 
print("--- %5.5s seconds ---" % (deltaTime))



print(NofFiles)


cv2.waitKey(0)
cv2.destroyAllWindows()