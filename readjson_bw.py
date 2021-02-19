import numpy as np 
import json
import os
import cv2

os.system("cls")

path0 = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\oregon_us\\"

path0 = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\CSC1202T\\"#ann
jsonPath = path0+"ann\\"#roads_annotated\\ds\\ann\\"
bwPath =   path0+"bw\\"#roads_annotated\\ds\\bw\\"
rgbPath =  path0+"img\\"#roads_annotated\\ds\\img\\"


rgbFiles = os.listdir(rgbPath)
img0 = cv2.imread(os.path.join(rgbPath,rgbFiles[5]))
M = img0.shape[0]
N = img0.shape[1]

jsonFiles = os.listdir(jsonPath)

for jF in jsonFiles:
#for idxF in range(3):
	#jF = jsonFiles[idxF]
	print(jF)

	with open(jsonPath+jF) as json_file:
		data = json.load(json_file)

	pcs = data['objects'][0]['points']['exterior']

	print(pcs)

	bwImg = np.zeros((M, N, 3), np.uint8)
	polygon = np.array(pcs).astype('int32')

	print(polygon)

	cv2.fillConvexPoly(bwImg, polygon, (255,255,255))

	#cv2.imshow(str(idxF),bwImg)
	

	cv2.imwrite((bwPath+jF[:-5]),bwImg)


cv2.waitKey(0)
cv2.destroyAllWindows()

