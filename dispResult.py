import cv2
import os
import numpy as np

rootPath = r"C:\Users\Esa\Pictures\_DATASET\unetpusbin\topCamSegmented\kFoldFolders\D0"
y = os.listdir(rootPath)

segPath = r"C:\Users\Esa\Pictures\_DATASET\unetpusbin\result_fold1" 

N = len(y)
# for i in range(N):
    # print(y[i])

idx = 170
rat = 0.2
rgbPath = os.path.join(rootPath,y[idx],"images",y[idx])
rgbPath = rgbPath+".png"
# print(rgbPath)
mskPath = os.path.join(rootPath,y[idx],"mask",y[idx])
mskPath = mskPath+".png"


segPath = os.path.join(segPath,y[idx])
segPath = segPath + ".png"




rgbImage = cv2.imread(rgbPath)
mskImage = cv2.imread(mskPath)
segImage = cv2.imread(segPath)

rgbImage = cv2.resize(rgbImage,[int(rgbImage.shape[1]*rat),int(rgbImage.shape[0]*rat)])
mskImage = cv2.resize(mskImage,[int(mskImage.shape[1]*rat),int(mskImage.shape[0]*rat)])
segImage = cv2.resize(segImage,[int(segImage.shape[1]*rat),int(segImage.shape[0]*rat)])
ovrImage = rgbImage.copy()

M = rgbImage.shape[0]
N = rgbImage.shape[1]

for i in range(M):
    for j in range(N):
        if (segImage[i,j,0]==255):
            ovrImage[i,j,0] = 0 




imgToDisp = np.hstack((rgbImage,mskImage))
imgToDisp = np.hstack((imgToDisp, segImage))
imgToDisp = np.hstack((imgToDisp, ovrImage))

cv2.imshow(y[idx]+".png", imgToDisp)

cv2.waitKey(0)
cv2.destroyAllWindows()