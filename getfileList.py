import cv2
import os
import matplotlib.pyplot as plt
import time

start_time = time.time()

os.system("cls")

# This the path for storing training data
path = "C:\\Users\\INKOM06\\Pictures\\roadlane-detection-evaluation-2013\\training\\"
#path = "C:\\Users\\INKOM06\\Pictures\\jagung2020\\largeDataSet\\true\\"
labelPath = path + "gt_image_2\\"
oriPath   = path + "image_2\\"

print(labelPath)
print(oriPath)


labFiles = []
oriFiles = []

#for idx in range(100):
# r=root, d=directories, f = files
#idx = 0
for r, d, f in os.walk(labelPath):
	for file in f:
		labFilesData = os.path.join(r, file)
		labFiles.append(labFilesData)
#		idx = idx + 1
#		print(str(idx)+":"+labFilesData)

nLab = (len(labFiles))

print("-----------------\n")
# r=root, d=directories, f = files
#idx = 0
for r, d, f in os.walk(oriPath):
	for file in f:
		oriFilesData = os.path.join(r, file)
		oriFiles.append(oriFilesData)
#		idx = idx + 1
#		print(str(idx)+":"+oriFilesData)

nOri = (len(oriFiles))

N = min(nLab, nOri)
NFin = N

for i in range(N):
	strOri = (str(i)+": "+oriFiles[i])
	strLab = (str(i)+": "+labFiles[i])
	oriFileName = (strOri[-13:])
	labFileName = (strLab[-18:])
	#print(str(i)+": "+oriFileName)
	#print(str(i)+": "+labFileName)
	if oriFileName[-10:] == labFileName[-10:]:
		print("====MATCH!!!====") 
	if oriFileName[-10:] != labFileName[-10:]:
		print("InEqual!==================================*****=========>>>") 
		NFin = NFin - 1
	print("-- -- -- ")

os.system("cls")

print("%d of %d are match "%(NFin,N))

savePath = path + "needtocheck\\"
savePath2 = path + "overlay\\"
savePath3 = path + "textfolder\\"


NofImg = min(NFin,N)

fOri = open(savePath3+"ori.txt","w")
fLab = open(savePath3+"lab.txt","w")

for idx in range(NofImg):
	print(oriFiles[idx])
	fileLoc = oriFiles[idx]
	fOri.write(fileLoc)
	fOri.write("\n")

	fileLoc = labFiles[idx]
	fLab.write(fileLoc)
	fLab.write("\n")


	#print(labFiles[idx])



fOri.close()
fLab.close()

deltaTime = time.time() - start_time 
print("--- %5.5s seconds ---" % (deltaTime))


cv2.waitKey()
cv2.destroyAllWindows()
