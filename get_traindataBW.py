import os
import cv2
import time

start_time = time.time()


os.system("cls")



bwPath = r'C:\Users\INKOM06\Pictures\roadDataset\CSC1202T\bw'
colPath = r'C:\Users\INKOM06\Pictures\roadDataset\CSC1202T\img'
savePath = r'C:\Users\INKOM06\Pictures\roadDataset\CSC1202T\_csv'

bwList  = os.listdir(bwPath)
colList = os.listdir(colPath)

bwFilePath  = os.path.join(bwPath,bwList[0])
colFilePath = os.path.join(colPath,colList[0])






print(bwPath)
print(colPath)

imgBW = cv2.imread(bwFilePath)
imgBW = cv2.cvtColor(imgBW, cv2.COLOR_BGR2GRAY)
ret,imgBW = cv2.threshold(imgBW,127,255,cv2.THRESH_BINARY)


imgCol = cv2.imread(colFilePath)


csvFileNm   = "oreS5_100img.csv"
csvFilePath = os.path.join(savePath,csvFileNm)
csvRoadFile = open(csvFilePath,"w+")
outVal = ("No, i, j,  r,  g, b, class")
csvRoadFile.write(outVal+"\n")



M = imgBW.shape[0]
N = imgBW.shape[1]
dataIdx = 0
deltaSamp = 5
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
		 #print("%d %d --- %d %d  %1.3f %1.3f  R:%1.2f  G:%1.2f  B:%1.2f"%(M,N,i,j,iR,jR,R,G,B))
		 outVal = "%d, %1.3f, %1.3f, %1.2f, %1.2f, %1.2f, %d"%(dataIdx,iR,jR,R,G,B,cl) 
		 #print("%1.3f, %1.3f, %1.2f, %1.2f, %1.2f"%(iR,jR,R,G,B))
		 print(outVal)

		 csvRoadFile.write(outVal+"\n")

		 dataIdx = dataIdx + 1



csvRoadFile.close()

deltaTime = time.time() - start_time 
print("--- %5.5s seconds ---" % (deltaTime))


#cv2.imshow("bw", imgBW)



cv2.waitKey(0)
cv2.destroyAllWindows()