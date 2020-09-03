import os
import cv2
import time
import math

start_time = time.time()

os.system("cls")

bwPath  = r'C:\Users\INKOM06\Pictures\roadDataset\oregon_us\roads_annotated\ds\bw'
colPath = r'C:\Users\INKOM06\Pictures\roadDataset\oregon_us\roads_annotated\ds\img'
savePath = r'C:\Users\INKOM06\Pictures\roadDataset\oregon_us\roads_annotated\ds\_csv'

bwList  = os.listdir(bwPath)
colList = os.listdir(colPath)

ratio = 0.3
deltaSamp = 20


bwFilePath  = os.path.join(bwPath,bwList[0])
imgBW = cv2.imread(bwFilePath)
imgBW = cv2.cvtColor(imgBW, cv2.COLOR_BGR2GRAY)
M = imgBW.shape[0]
N = imgBW.shape[1]
m = math.floor(M*ratio)
n = math.floor(N*ratio)



NofFiles = len(bwList)
idxOfTrainFile = 0

bwFilePath  = os.path.join(bwPath,bwList[idxOfTrainFile])
colFilePath = os.path.join(colPath,colList[idxOfTrainFile])





imgBW = cv2.imread(bwFilePath)
imgBW = cv2.cvtColor(imgBW, cv2.COLOR_BGR2GRAY)
ret,imgBW = cv2.threshold(imgBW,127,255,cv2.THRESH_BINARY)

imgCol = cv2.imread(colFilePath)


# Resize both images
imgBW  = cv2.resize(imgBW, (n,m), interpolation = cv2.INTER_AREA) 
imgCol = cv2.resize(imgCol, (n,m), interpolation = cv2.INTER_AREA) 



csvFileNm   = "oreS5_100img.csv"
csvFilePath = os.path.join(savePath,csvFileNm)
csvRoadFile = open(csvFilePath,"w+")
outVal = ("No, i, j,  r,  g, b, class")
csvRoadFile.write(outVal+"\n")



M = imgBW.shape[0]
N = imgBW.shape[1]
dataIdx = 0
for i in range(0,M,deltaSamp):
	for j in range(0,N,deltaSamp):
		 if imgBW[i,j] == 255:
		 	cl = 1
		 if imgBW[i,j] == 0:
		 	cl = 0

		 iR = i/M
		 jR = j/N
		 R = imgCol[i,j,2]/255
		 G = imgCol[i,j,1]/255
		 B = imgCol[i,j,0]/255
		 outVal = "%d, %1.3f, %1.3f, %1.2f, %1.2f, %1.2f, %d"%(dataIdx,iR,jR,R,G,B,cl) 
		 print("%d %d  ---- %s"%(i,j,outVal))

		 csvRoadFile.write(outVal+"\n")

		 dataIdx = dataIdx + 1



csvRoadFile.close()

deltaTime = time.time() - start_time 
print("--- %5.5s seconds ---" % (deltaTime))



print(NofFiles)

cv2.waitKey(0)
cv2.destroyAllWindows()