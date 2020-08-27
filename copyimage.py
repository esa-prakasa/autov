import numpy as np
import cv2
import os


os.system("cls")

path = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\"
path0 = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\oregon_us\\"
path1 = path0+"rgb\\"
path2 = path0+"rgbtrain\\"


print(path1)
print(path2)


files = os.listdir(path1)

frameIdx = 0
for fileName in files:
	
	if (frameIdx%10 == 0):
		print(str(frameIdx)+" "+fileName)
		img = cv2.imread(path1+fileName)
		cv2.imwrite(path2+fileName,img)
	frameIdx +=1