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

deltaTime = time.time() - start_time 
print("--- %5.5s seconds ---" % (deltaTime))


cv2.waitKey()
cv2.destroyAllWindows()
