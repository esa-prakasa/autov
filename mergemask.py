import os
import cv2
import numpy as np


os.system('cls')
path =r'C:\Users\INKOM06\Pictures\_DATASET\Unet\stage0_test\fc5452f612a0f972fe55cc677055ede662af6723b5c1615ad539b8a4bd279bdb\masks'
path = r'C:\Users\INKOM06\Pictures\_DATASET\Unet\stage0_test\fc9269fb2e651cd4a32b65ae164f79b0a2ea823e0a83508c85d7985a6bed43cf\masks'
path = r'C:\Users\INKOM06\Pictures\_DATASET\Unet\stage0_test\fd8065bcb1afdbed19e028465d5d00cd2ecadc4558de05c6fa28bea3c817aa22\masks'
path = r'C:\Users\INKOM06\Pictures\_DATASET\Unet\stage0_test\fdda64c47361b0d1a146e5b7b48dc6b7de615ea80b31f01227a3b16469589528\masks'
path = r'C:\Users\INKOM06\Pictures\_DATASET\Unet\stage0_test\fe80a2cf3c93dafad8c364fdd1646b0ba4db056cdb7bdb81474f957064812bba\masks'
path = r'C:\Users\INKOM06\Pictures\_DATASET\Unet\stage0_test\fec226e45f49ab81ab71e0eaa1248ba09b56a328338dce93a43f4044eababed5\masks'

path = r'C:\Users\INKOM06\Pictures\_DATASET\Unet\stage0_test\feffce59a1a3eb0a6a05992bb7423c39c7d52865846da36d89e2a72c379e5398\masks'
path = r'C:\Users\INKOM06\Pictures\_DATASET\Unet\stage0_test\ff3e512b5fb860e5855d0c05b6cf5a6bcc7792e4be1f0bdab5a00af0e18435c0\masks'
path = r'C:\Users\INKOM06\Pictures\_DATASET\Unet\stage0_test\ff599c7301daa1f783924ac8cbe3ce7b42878f15a39c2d19659189951f540f48\masks'
path = r'C:\Users\INKOM06\Pictures\_DATASET\Unet\stage0_test\ff3407842ada5bc18be79ae453e5bdaa1b68afc842fc22fa618ac6e6599d0bb3\masks'


maskList = os.listdir(path)
NF = len(maskList)

img = cv2.imread(os.path.join(path,maskList[0]))
M = img.shape[0]
N = img.shape[1]

mask = np.zeros((M,N), dtype = np.uint8)

print("Number of file %d Dimension %d x %d"%(NF,M,N))

for i in range(NF):
	print("%d %s"%(i,maskList[i]))
	img = cv2.imread(os.path.join(path,maskList[i]))
	mask = mask + img[:,:,0]


cv2.imshow("Final mask", mask)



cv2.imwrite(os.path.join(path,'_mask.png'), mask)

#for mFile in maskList:
#	print(mFile)

cv2.waitKey(0)
cv2.destroyAllWindows()