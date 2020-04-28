import cv2
import os
import matplotlib.pyplot as plt
import time

start_time = time.time()

os.system("cls")



labelPath ="C:\\Users\\INKOM06\\Pictures\\roadlane-detection-evaluation-2013\\data_road\\training\\gt_image_2\\"
oriPath ="C:\\Users\\INKOM06\\Pictures\\roadlane-detection-evaluation-2013\\data_road\\training\\image_2\\"
#pathToSave = "C:\\Users\\INKOM06\\Documents\\[0--KEGIATAN-Ku-2020\\2020.01-006-Autonomous Vehicle Project\\pixSegmentation202004\\roadreg\\"

#fileTxtName = "rat30pct0414.txt"
#nonRoadName = "nonroad30pct_0414.txt" 
#NofImage = 1 #200
idxSc = 215

ratio = 0.2
ratioPct = int(ratio*100)

#csvRoadFileNm = "test_road"+str(ratioPct)+"pct"+"_"+str(idxSc)+".csv"


#logFile = open((pathToSave+fileTxtName),"w+")
#logFileNR = open((pathToSave+nonRoadName),"w+")
#csvRoadFile = open((pathToSave+"csvtest\\"+csvRoadFileNm),"w+")



#outVal = ("No, i, j,  r,  g, b, class")
#csvRoadFile.write(outVal+"\n")



#for idx in range(idxSc,(idxSc+20),1):
for idx in range(1,280,1):
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
	print(str(idx)+"  "+oriFiles[idx])


	#if (idx>210):
	#	print(str(idx)+"  "+oriFiles[idx])
