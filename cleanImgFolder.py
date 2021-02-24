import cv2
import os


os.system("cls")

bwPath = r'C:\Users\INKOM06\Pictures\roadDataset\CSC1202T\bw'
imgPath = r'C:\Users\INKOM06\Pictures\roadDataset\CSC1202T\img'
imgCPath = r'C:\Users\INKOM06\Pictures\roadDataset\CSC1202T\imgc'

bwList  = os.listdir(bwPath)
imgList = os.listdir(imgPath)

NBw = len(bwList)

NImg = len(imgList)

idx = 0
for x in bwList:
	print(str(idx)+"  "+bwList[idx])
	fullPathRd = os.path.join(imgPath, bwList[idx])
	fullPathSv = os.path.join(imgCPath, bwList[idx])

	I = cv2.imread(fullPathRd)
	cv2.imwrite(fullPathSv,I)


	print(fullPathSv)
	idx = idx + 1




cv2.waitKey(0)
cv2.destroyAllWindows()