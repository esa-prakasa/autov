import os  
import json 
import cv2
import numpy as np
import random as rd

os.system("cls")

annPath = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\Purbaleunyi_Highway\\ds\\ann\\"
imgPath = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\bandung\\Purbaleunyi_Highway\\ds\\img\\"

annFiles = os.listdir(annPath)
imgFiles = os.listdir(imgPath)

N = (len(annFiles))

#for i in range(N):
#	print(annFiles[i]+" --- "+imgFiles[i])

#fileIdx = int(input("Give file index (max 40) ")) 

fileIdx = rd.randrange(40)


# Opening JSON file 
f = open(annPath+annFiles[fileIdx])
img = cv2.imread(imgPath+imgFiles[fileIdx]) 

blackImg = np.zeros((img.shape[0],img.shape[1]), dtype =np.uint8)


data = json.load(f) 
 
color = (0, 255, 0)
whiteCol = (255, 255, 255)
thickness = 8
whiteThickness = 2

NPoints = (len(data['objects']))   # number of lines 

for i in range(NPoints): 
    lineData = data['objects'][i]['points']['exterior']
    print(data['objects'][i]['points']['exterior'])
    N = (len(lineData))
    strIdx = 0
    for j in range(1,N):
    	print(lineData[j])    	
    	endIdx = j
    	x  =  lineData[strIdx][0]
    	y   = lineData[strIdx][1]
    	start_point = (x,y)

    	x  =  lineData[endIdx][0]
    	y   = lineData[endIdx][1]
    	end_point = (x, y) #lineData[1]

    	img = cv2.line(img, start_point, end_point, color, thickness)
    	blackImg = cv2.line(blackImg, start_point, end_point, whiteCol, whiteThickness) 

    	strIdx = endIdx
 
cv2.imshow("img", img) 
f.close() 

cv2.imshow("anno", blackImg) 

#img2 = img.copy()

#li = cv2.InitLineIterator(img, start_point, end_point)
#print(len(li))





cv2.waitKey(0)
cv2.destroyAllWindows()
