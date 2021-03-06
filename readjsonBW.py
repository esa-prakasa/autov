import numpy as np 
import json
import os
import cv2

os.system("cls")

path0 = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\oregon_us\\"
jsonPath = path0+"roads_annotated\\ds\\ann\\"
bwPath = path0+"roads_annotated\\ds\\bw\\"
rgbPath = path0+"roads_annotated\\ds\\img\\"

jsonFiles = os.listdir(jsonPath)

for jF in jsonFiles:
	print(jF)

	with open(jsonPath+jF) as json_file:
		data = json.load(json_file)

	pcs = data['objects'][0]['points']['exterior']

	bwImg = np.zeros((324, 576, 3), np.uint8)
	polygon = np.array(pcs)
 
	cv2.fillConvexPoly(bwImg, polygon, (255, 255, 255))
	cv2.imwrite((bwPath+jF[:-5]),bwImg)


cv2.waitKey(0)
cv2.destroyAllWindows()

