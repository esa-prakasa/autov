import numpy as np 
import json
import os
import cv2

os.system("cls")

jsonPath = r"C:\Users\Esa\Documents\___KEGIATAN-Ku 2021\00.00 -- Autonomous Vehicle\edgeLine\00008.png.json"
rgbPath =  r"C:\Users\Esa\Documents\___KEGIATAN-Ku 2021\00.00 -- Autonomous Vehicle\edgeLine\PusbinTopCam_topCamFrames_00008.png"


img = cv2.imread(rgbPath)
M = img.shape[0]
N = img.shape[1]

bwImg = np.zeros((M, N, 3), np.uint8)


with open(jsonPath) as json_file:
	data = json.load(json_file)

edge1 = data['objects'][1]['points']['exterior']
edge2 = data['objects'][2]['points']['exterior']

print(edge1)
print(edge2)


for kEdge in range(1,3,1):
	edge = data['objects'][kEdge]['points']['exterior']
	N1 = len(edge)

	for k in range(0,(N1-1),1):
		start_point = (edge[k][0], edge[k][1]) 
		end_point = (edge[k+1][0], edge[k+1][1])
		color = (0, 255, 0)
		wtColor = (255, 255, 255)
		thickness = 5
		img = cv2.line(img, start_point, end_point, color, thickness)
		bwImg = cv2.line(bwImg, start_point, end_point, wtColor, thickness)

M = img.shape[0]
N = img.shape[1]

ratio =0.3


img = cv2.resize(img, (round(ratio*N),round(ratio*M)), interpolation = cv2.INTER_AREA)
bwImg = cv2.resize(bwImg, (round(ratio*N),round(ratio*M)), interpolation = cv2.INTER_AREA)

finalImg = np.hstack((img,bwImg))

#cv2.imshow("RGB",img)
#cv2.imshow("BW",bwImg)
cv2.imshow("RGB and Reference Image", finalImg)


cv2.waitKey(0)
cv2.destroyAllWindows()


