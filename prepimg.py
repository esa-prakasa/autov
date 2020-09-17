import os
import cv2

imgPath = r'C:\Users\INKOM06\Pictures\roadDataset\oregon_us\rgb'
imgFiles = os.listdir(imgPath)
#imgIdx  = int(input("Which image index that you want? Max: "))
imgIdx = 5
imgPath = os.path.join(imgPath,imgFiles[imgIdx])
print(imgPath)


img = cv2.imread(imgPath)


ratio = 0.3

M = img.shape[0]
N = img.shape[1]


img = cv2.resize(img, (int(N*ratio),int(M*ratio)),interpolation = cv2.INTER_AREA)

M = img.shape[0]
N = img.shape[1]


for i in range(M):
	for j in range(N):
		#print("%d  %d  %d  %d  %d"%(i,j,img[i,j,2],img[i,j,1],img[i,j,0]))
		print("%f  %f  %f  %f  %f"%(i/M,j/N,img[i,j,2]/255,img[i,j,1]/255,img[i,j,0]/255))





cv2.imshow("img", img)


cv2.waitKey(0)
cv2.destroyAllWindows()


