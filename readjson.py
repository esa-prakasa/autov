import numpy as np 
import json
import os
import cv2

path0 = "C:\\Users\\INKOM06\\Pictures\\roadDataset\\oregon_us\\"
path3 = path0+"jsonpoly\\"

files = os.listdir(path3)

#print(files[0])


with open(path3+files[0]) as json_file:
    data = json.load(json_file)


#print(data)

pcs = data['objects'][0]['points']['exterior']
N = len(pcs)

for i in range (len(pcs)):
	print(pcs[i])


img = np.zeros((324, 576, 3), np.uint8)
triangle = np.array(pcs)
 
cv2.fillConvexPoly(img, triangle, (255, 255, 255))
cv2.imshow("BW",img)


cv2.waitKey(0)
cv2.destroyAllWindows()

